from pathlib import Path
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import json

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compliance.tpl"
with open(PROMPT_PATH, "r") as f:
    COMP_PROMPT = f.read()

def check_compliance(weights: dict, rules: dict, llm_model="gpt-4o-mini"):
    """
    weights: dict asset_class -> weight
    rules: dict with keys max_single_asset, required_cash_buffer
    """
    input_json = json.dumps({"weights": weights, "rules": rules})
    prompt = COMP_PROMPT.replace("{input_json}", input_json)
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    out = chain.run({})
    try:
        parsed = json.loads(out)
    except Exception:
        # fallback deterministic checks
        violations = []
        for k,v in weights.items():
            if v > rules.get("max_single_asset", 0.5):
                violations.append(f"{k} weight > max_single_asset")
        if weights.get("cash", 0.0) < rules.get("required_cash_buffer", 0.0):
            violations.append("cash below required buffer")
        parsed = {"compliant": len(violations)==0, "violations": violations}
    return parsed
