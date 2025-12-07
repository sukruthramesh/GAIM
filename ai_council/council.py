import time, json, statistics
import numpy as np
from ai_council.generate_prompts import *
from langchain_ollama import OllamaLLM

MODELS = [
    {"id" : "expert_1", "llm" : OllamaLLM(model="gpt-oss:20b"), "name" : "gpt-oss:20b"},
    {"id" : "expert_2", "llm" : OllamaLLM(model="deepseek-r1:8b"), "name" : "deepseek-r1:8b"},
    {"id" : "expert_3", "llm" : OllamaLLM(model="ministral-3"), "name" : "minstral-3"},
    {"id" : "expert_4", "llm" : OllamaLLM(model="mistral:7b"), "name" : "mistral:7b"},
    {"id" : "expert_5", "llm" : OllamaLLM(model="phi3:mini"), "name" : "phi3:mini"},      ### Makes mistakes and brings for variance
    {"id" : "expert_6", "llm" : OllamaLLM(model="gemma2:9b"), "name": "gemma2:9b"},
    {"id" : "evaluator", "llm" : OllamaLLM(model="starling-lm"), "name" : "starling-1m"}
]

WEIGHTS = {
    'accuracy' : 0.35,
    'completeness' : 0.25,
    'grounding' : 0.20,
    'reasoning' : 0.15,
    'clarity' : 0.05
}

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