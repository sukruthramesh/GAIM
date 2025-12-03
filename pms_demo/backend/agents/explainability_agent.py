from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import json

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "explainability.tpl"
with open(PROMPT_PATH, "r") as f:
    EXPLAIN_PROMPT = f.read()

def explain_allocation(payload: dict, llm_model="gpt-4o-mini"):
    prompt = EXPLAIN_PROMPT.replace("{input_json}", json.dumps(payload))
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    out = chain.run({})
    return out
