from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import json
from ..utils import compute_A

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "q11_semantic.tpl"
with open(PROMPT_PATH, "r") as f:
    Q11_PROMPT = f.read()

def semantic_score_q11(q11_text: str, llm_model="gpt-4o-mini"):
    prompt = Q11_PROMPT.replace("{q11_text}", q11_text.replace('"','\\"'))
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    resp = chain.run({})
    try:
        parsed = json.loads(resp)
        score = float(parsed.get("q11_score", 2.0))
        rationale = parsed.get("rationale","")
    except Exception:
        score = 2.0
        rationale = ""
    return score, rationale

def compute_risk_coefficient(q_mandatory, q11_text=None, q12_s=None, include_optionals=True):
    q11_score = None
    rationale = None
    if q11_text:
        q11_score, rationale = semantic_score_q11(q11_text)
    A, meta = compute_A(q_mandatory, q11=q11_score, q12_s=q12_s, include_optionals=include_optionals)
    meta.update({"q11_rationale": rationale})
    return A, meta
