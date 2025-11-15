import json
from typing import Dict
import openai
from dataclasses import dataclass

@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str

    def run(self, user_message, temperature = 0.0) ->str:
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_message}
        ]
        resp = openai.ChatCompletion.create(
            model = self.model,
            messages = messages,
            temperature = temperature,
            max_tokens = 1000
        )
        return resp["choices"][0]["message"]["content"]
    
def initialise_prompts(filename):
    with open(filename, 'r') as file:
        file_content = file.read()
    prompts = file_content.split("<break>")
    return prompts
    
PROMPT_FILE_PATH = "Prompts\Prompts_V1.txt"

(
    PROBLEM_NORMALIZER_PROMPT,
    ENUMERATOR_PROMPT,
    EVALUATOR_PROMPT,
    SELECTOR_PROMPT,
    AUDITOR_PROMPT,
    VALIDATOR_PROMPT,
) = initialise_prompts(PROMPT_FILE_PATH)