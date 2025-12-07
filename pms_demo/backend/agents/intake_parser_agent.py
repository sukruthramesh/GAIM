from langchain import LLMChain, PromptTemplate # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore
import json
from pathlib import Path
import os


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "intent_parser.tpl"
with open(PROMPT_PATH, "r") as f:
    PROMPT_TEMPLATE = f.read()

def parse_intent(q11_text: str, llm_model=None):
    if not q11_text:
        return {"goal_type":"wealth_creation","horizon_years":5,"constraints":{}}
    model = llm_model or (os.environ.get("OPENAI_MODEL") if "OPENAI_MODEL" in globals() else "gpt-4o-mini")
    prompt = PROMPT_TEMPLATE.replace("{user_text}", q11_text.replace('"','\\"'))
    llm = ChatOpenAI(temperature=0.0, model=model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    resp = chain.run({})
    try:
        parsed = json.loads(resp)
    except Exception:
        parsed = {"goal_type":"wealth_creation","horizon_years":5,"constraints":{}}
    return parsed
