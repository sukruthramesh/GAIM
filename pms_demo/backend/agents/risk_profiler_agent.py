# backend/agents/risk_profiler_agent.py
import os
import json
from pathlib import Path
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Using the LCEL-style pipeline (prompt | llm)
# Newer LangChain versions support `prompt | llm` and .invoke().

from utils import compute_A

# Load the template file (must exist at backend/prompts/q11_semantic.tpl)
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "q11_semantic.tpl"
if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH}")
Q11_TEMPLATE_TEXT = PROMPT_PATH.read_text(encoding="utf-8")

def _make_pipeline(model_name: Optional[str] = None, temperature: float = 0.0):
    """
    Build a prompt -> model pipeline. Returns a callable pipeline that accepts the input dict.
    """
    model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(Q11_TEMPLATE_TEXT)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    pipeline = prompt | llm
    return pipeline

def semantic_score_q11(q11_text: str, llm_model: Optional[str] = None) -> Tuple[Optional[float], str]:
    """
    Convert a free-form risk description into a numeric score 0..4 using an LLM.
    Returns (score, rationale). If LLM fails, returns (None, "").
    """
    if not q11_text:
        return None, ""

    pipeline = _make_pipeline(model_name=llm_model)
    # invoke the pipeline with the variable used in the template: {q11_text}
    try:
        # .invoke returns the model output as a string in modern LCEL pipelines
        out = pipeline.invoke({"q11_text": q11_text})
        # ensure we have a string
        if not isinstance(out, str):
            out = str(out)

        # Try to extract a JSON object from the response (safe parse)
        # Many LLMs sometimes add commentary; extract first {...}
        parsed = None
        try:
            parsed = json.loads(out)
        except Exception:
            # try to extract JSON substring
            import re
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        if parsed is None:
            # fallback: attempt to parse simple "q11_score: X" pattern
            import re
            m = re.search(r"q11_score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", out, flags=re.IGNORECASE)
            if m:
                score = float(m.group(1))
                rationale = ""
                return max(0.0, min(4.0, score)), rationale
            # last resort: return None
            return None, out.strip()[:200]

        score = parsed.get("q11_score", None)
        if score is None:
            # try alt keys
            for k in ["score", "q11", "q11_score"]:
                if k in parsed:
                    score = parsed[k]
                    break

        rationale = parsed.get("rationale", "") if isinstance(parsed.get("rationale", ""), str) else ""

        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        if score_f is None:
            return None, rationale
        # clip
        score_f = max(0.0, min(4.0, score_f))
        return score_f, rationale

    except Exception as e:
        # On any LLM/pipeline error, return safe defaults
        return None, f"LLM error: {repr(e)}"

def compute_risk_coefficient(q_mandatory, q11_text: Optional[str] = None, q12_self: Optional[int] = None,
                             include_optionals: bool = True) -> Tuple[float, dict]:
    """
    Compute the final risk-aversion coefficient A and return (A, meta).
    q_mandatory: list of 10 ints (0..4)
    q11_text: optional free-text description
    q12_self: optional self-rated 1..10
    include_optionals: whether to include q11/q12 in scoring
    """
    q11_score = None
    q11_rationale = ""
    if include_optionals and q11_text:
        q11_score, q11_rationale = semantic_score_q11(q11_text)

    # compute A using backend.utils.compute_A
    A, meta = compute_A(q_mandatory, q11=q11_score, q12_s=q12_self, include_optionals=include_optionals)
    meta.update({"q11_rationale": q11_rationale})
    return A, meta
