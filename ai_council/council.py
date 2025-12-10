import time, json, statistics, ast, re
import numpy as np
# from ai_council.generate_prompts import *
from ai_council.prompts import *
from ai_council.constants import *
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

scoring_results = []

def extract_first_curly_balanced(text):
    """
    Extracts the substring between the first balanced set of curly braces {}.
    Handles nested braces correctly.
    Returns None if no valid balanced braces are found.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    start_index = None
    brace_count = 0

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_index = i + 1  # content starts after '{'
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_index is not None:
                return text[start_index:i]  # return content inside braces
            if brace_count < 0:
                # Found closing brace before opening
                return None

    return None  # No balanced braces found



def pretty_print_json(json_data):
    """
    Pretty prints JSON data with indentation and sorted keys.
    Accepts either a Python dict/list or a JSON string.
    """
    try:
        # If input is a string, parse it to Python object
        if isinstance(json_data, str):
            json_obj = json.loads(json_data)
        elif isinstance(json_data, (dict, list)):
            json_obj = json_data
        else:
            raise TypeError("Input must be a JSON string, dict, or list.")

        # Pretty print with indentation and sorted keys
        pretty_json = json.dumps(json_obj, indent=4, sort_keys=True, ensure_ascii=False)
        print(pretty_json)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON string: {e}")
    except Exception as e:
        print(f"Error: {e}")


# def generate_scores(responses, user_prompt, llm = None):
#     scorer_parser = PydanticOutputParser(pydantic_object=scoring_output)
#     scoring_prompt = ChatPromptTemplate.from_template(scoring_template)
#     chains = [(scoring_prompt|k["llm"], k) for k in MODELS]
#     scoring_matrix = {}
#     for chain in chains:
#         if chain[1]['id'] == "evaluator":
#             continue
#         if llm and chain[1]['name'] != llm:
#             continue
#         temp_dict = {}
#         print(chain[1]['name'])
#         for i in range(len(responses)):
#             result = chain[0].invoke({"user_prompt" : user_prompt, "candidate_response" : responses[i], "output_format" : scorer_parser.get_format_instructions()})
#             try:
#                 scoring_results.append(result)
#                 json_response = ast.literal_eval('{'+re.sub(r'//.*','',extract_first_curly_balanced(scoring_results[-1])).replace('null' , '0') + '}')
#                 json_response['total'] = sum([WEIGHTS[k] * json_response['scores'][k] for k in WEIGHTS])
#                 temp_dict[responses[i]["response_id"]] = json_response
#                 print(f"Scoring complete for {responses[i]['response_id']} by {chain[1]['name']}")
#             except Exception as e:
#                 print(f"Could not parse {responses[i]['response_id']} by {chain[1]['name']} due to {e}")
#         scoring_matrix[chain[1]['name']] = temp_dict
#     return scoring_matrix



import ast
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def generate_scores(responses, user_prompt, llm=None, timeout_seconds=600, max_workers=4):
    """
    Score candidate responses using multiple LLM chains with a timeout on each invoke.
    - timeout_seconds: how long to wait for chain.invoke before skipping that response.
    - max_workers: threadpool size for wrapping blocking invoke() calls.
    """
    scorer_parser = PydanticOutputParser(pydantic_object=scoring_output)
    scoring_prompt = ChatPromptTemplate.from_template(scoring_template)
    chains = [(scoring_prompt | k["llm"], k) for k in MODELS]

    scoring_matrix = {}
    scoring_results = []  # collected raw results (same as your previous usage)

    # Reuse a threadpool to wrap blocking invoke() calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chain in chains:
            # skip evaluator and optionally filter by llm name
            if chain[1]["id"] == "evaluator":
                continue
            if llm and chain[1]["name"] != llm:
                continue

            temp_dict = {}
            print(chain[1]["name"])

            for i in range(len(responses)):
                payload = {
                    "user_prompt": user_prompt,
                    "candidate_response": responses[i],
                    "output_format": scorer_parser.get_format_instructions()
                }

                # submit the blocking invoke() to the threadpool and wait with timeout
                future = executor.submit(chain[0].invoke, payload)
                try:
                    result = future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    # timed out — continue the loop
                    print(f"Timeout after {timeout_seconds}s for {responses[i]['response_id']} by {chain[1]['name']}")
                    # Optionally: future.cancel()  # best-effort (may not have effect if already running)
                    continue
                except Exception as e:
                    # other errors from invoke()
                    print(f"Invoke error for {responses[i]['response_id']} by {chain[1]['name']}: {e}")
                    continue

                # Parsing and weighting (kept your original logic, with minor safety)
                try:
                    scoring_results.append(result)
                    raw = extract_first_curly_balanced(scoring_results[-1])
                    cleaned = re.sub(r'//.*', '', raw).replace('null', '0')
                    json_response = ast.literal_eval("{" + cleaned + "}")
                    json_response["total"] = sum(WEIGHTS[k] * json_response["scores"][k] for k in WEIGHTS)
                    temp_dict[responses[i]["response_id"]] = json_response
                    print(f"Scoring complete for {responses[i]['response_id']} by {chain[1]['name']}")
                except Exception as e:
                    print(f"Could not parse {responses[i]['response_id']} by {chain[1]['name']} due to {e}")

            scoring_matrix[chain[1]["name"]] = temp_dict

    return scoring_matrix




def generate_expert_response(user_prompt, context):
    prompt = ChatPromptTemplate.from_template(expert_generation_template)
    return_prompt = prompt.invoke({"user_prompt" : user_prompt, "context" : context})
    chains = [(prompt|k['llm'] , k) for k in MODELS]
    responses = []
    for i, chain in enumerate(chains):
        if chain[1]['id'] == "evaluator":
            continue
        print(f"Response from {chain[1]['name']}")
        result = chain[0].invoke({"user_prompt" : user_prompt, "context" : context})
        responses.append(
            {
                "response_id" : f"r_{i}",
                "model_id" : chain[1]["id"],
                "text" : result
            }
        )
        print(f"Response generated by {chain[1]['name']} with id r_{i}")
    return responses, return_prompt

def generate_audit_report(user_prompt, responses, scoring_matrix):
    audit_report = PydanticOutputParser(pydantic_object=Audit_Report)
    audit_prompt = ChatPromptTemplate.from_template(auditor_prompt_template)
    auditor =[k for k in MODELS if k["id"] == "evaluator"]
    chain = audit_prompt|auditor[0]['llm']
    # print(audit_prompt)
    result = chain.invoke({"user_prompt" : user_prompt, "responses" : responses, "scoring_matrix" : scoring_matrix, "output_format": audit_report.get_format_instructions()})
    print(result)
    return result, audit_prompt

def print_with_bold(text):
    """
    Prints text with **bold** markers converted to terminal bold.
    Example: "This is **bold** text" -> This is bold text (in bold)
    """
    BOLD_START = "\033[1m"
    BOLD_END = "\033[0m"
    if not isinstance(text, str):
        print("Error: Input must be a string.")
        return

    # Replace **...** with ANSI bold codes
    formatted_text = re.sub(
        r"\*\*(.*?)\*\*",  # Match text between ** and **
        lambda m: f"{BOLD_START}{m.group(1)}{BOLD_END}",
        text
    )
    print(formatted_text)

def compute_average_totals(scoring_matrix):
    totals = {}      # collects all totals per response_id
    counts = {}      # counts how many values per response_id
    for model_data in scoring_matrix.values():
        for response_id, response_content in model_data.items():
            total_value = response_content["total"]
            if response_id not in totals:
                totals[response_id] = 0
                counts[response_id] = 0
            totals[response_id] += total_value
            counts[response_id] += 1
    averages = {response_id: totals[response_id] / counts[response_id]
                for response_id in totals}
    return averages


def audited_scoring_matrix(audit , scoring_matrix, responses):
    response_ids = [k['response_id'] for k in responses]
    audit_json = ast.literal_eval('{'+extract_first_curly_balanced(audit)+"}")
    for model in audit_json['normalization']:
        for j in scoring_matrix[model]:
            scoring_matrix[model][j]['total'] *= audit_json['normalization'][model]
    for drops in audit_json['drops']:
        scoring_matrix.pop(drops)
    averages_json = compute_average_totals(scoring_matrix)
    best_response = responses[response_ids.index(max(averages_json))]
    return scoring_matrix, averages_json, best_response
