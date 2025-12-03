from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import json
from pathlib import Path

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "intent_parser.tpl"
with open(PROMPT_PATH, "r") as f:
    PROMPT_TEMPLATE = f.read()

def parse_intent(q11_text: str, llm_model="gpt-4o-mini"):
    prompt = PROMPT_TEMPLATE.replace("{user_text}", q11_text.replace('"','\\"'))
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    resp = chain.run({})
    # the prompt ensures JSON output - parse it
    try:
        parsed = json.loads(resp)
    except Exception:
        # fallback - basic
        parsed = {"goal_type":"wealth_creation","horizon_years":5,"constraints":{}}
    return parsed
