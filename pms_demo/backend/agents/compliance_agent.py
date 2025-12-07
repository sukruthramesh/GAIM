from pathlib import Path
from langchain import LLMChain, PromptTemplate # pyright: ignore[reportMissingImports]
from langchain.chat_models import ChatOpenAI # type: ignore
import json
import os

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compliance.tpl"
with open(PROMPT_PATH, "r") as f:
    COMP_PROMPT = f.read()

def check_compliance(weights: dict, rules: dict, llm_model=None):
    input_json = json.dumps({"weights": weights, "rules": rules})
    model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = COMP_PROMPT.replace("{input_json}", input_json)
    llm = ChatOpenAI(temperature=0.0, model=model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    out = chain.run({})
    try:
        parsed = json.loads(out)
    except Exception:
        violations = []
        for k,v in weights.items():
            if v > rules.get("max_single_asset", 0.5):
                violations.append(f"{k} weight > max_single_asset")
        if weights.get("cash", 0.0) < rules.get("required_cash_buffer", 0.0):
            violations.append("cash below required buffer")
        parsed = {"compliant": len(violations)==0, "violations": violations}
    return parsed
