from langchain import LLMChain, PromptTemplate # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore
from pathlib import Path
import json
import os

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "explainability.tpl"
with open(PROMPT_PATH, "r") as f:
    EXPLAIN_PROMPT = f.read()

def explain_allocation(payload: dict, llm_model=None):
    model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = EXPLAIN_PROMPT.replace("{input_json}", json.dumps(payload))
    llm = ChatOpenAI(temperature=0.0, model=model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    out = chain.run({})
    return out
