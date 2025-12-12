from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
import os



# MODELS = [
#     {"id" : "expert_1", "llm" : OllamaLLM(model="deepseek-r1:8b"), "name" : "deepseek-r1:8b"},
#     {"id" : "expert_2", "llm" : OllamaLLM(model="starling-lm"), "name" : "starling-1m"},
#     {"id" : "expert_3", "llm" : OllamaLLM(model="ministral-3"), "name" : "minstral-3"},
#     {"id" : "expert_4", "llm" : OllamaLLM(model="mistral:7b"), "name" : "mistral:7b"},
#     {"id" : "expert_5", "llm" : OllamaLLM(model="phi3:mini"), "name" : "phi3:mini"},      ### Makes mistakes and brings for variance
#     {"id" : "evaluator", "llm" : OllamaLLM(model="gemma2:9b"), "name": "gemma2:9b"},
#     {"id" : "expert_6", "llm" : OllamaLLM(model="gpt-oss:20b"), "name" : "gpt-oss:20b"}
# ]

IS_ONLINE = False

ONLINE_MODLES =[
    {"id" : "expert_1", "llm" : ChatOpenAI(model="gpt-4.1-nano"), "name" : "gpt-4.1-nano"},
    {"id" : "expert_2", "llm" : ChatOpenAI(model="gpt-4o-mini"), "name" : "gpt-4o-mini"},
    {"id" : "expert_3", "llm" : ChatOpenAI(model="gpt-4.1-mini"), "name" : "gpt-4.1-mini"},
    # {"id" : "expert_4", "llm" : ChatOpenAI(model="mistral:7b"), "name" : "mistral:7b"},
    # {"id" : "expert_5", "llm" : ChatOpenAI(model="phi3:mini"), "name" : "phi3:mini"},      ### Makes mistakes and brings for variance
    {"id" : "evaluator", "llm" : ChatOpenAI(model="gpt-5-mini"), "name": "gpt-5-mini"},
    # {"id" : "expert_6", "llm" : ChatOpenAI(model="gpt-oss:20b"), "name" : "gpt-oss:20b"}
]

OFFLINE_MODELS = [
    # {"id" : "expert_1", "llm" : OllamaLLM(model="deepseek-r1:8b"), "name" : "deepseek-r1:8b"},
    {"id" : "expert_2", "llm" : OllamaLLM(model="starling-lm"), "name" : "starling-1m"},
    {"id" : "expert_3", "llm" : OllamaLLM(model="ministral-3"), "name" : "minstral-3"},
    {"id" : "expert_4", "llm" : OllamaLLM(model="mistral:7b"), "name" : "mistral:7b"},
    {"id" : "expert_5", "llm" : OllamaLLM(model="phi3:mini"), "name" : "phi3:mini"},      ### Makes mistakes and brings for variance
    {"id" : "evaluator", "llm" : OllamaLLM(model="gemma2:9b"), "name": "gemma2:9b"},
    # {"id" : "expert_6", "llm" : OllamaLLM(model="gpt-oss:20b"), "name" : "gpt-oss:20b"}
]

MODELS = ONLINE_MODLES if IS_ONLINE else OFFLINE_MODELS

WEIGHTS = {
    'accuracy' : 0.35,
    'completeness' : 0.25,
    'grounding' : 0.20,
    'reasoning' : 0.15,
    'clarity' : 0.05
}

DATA_DOC_FOLDER = "ai_council/Docs"
VECTOR_DB_FOLDER = "ai_council/Vectors/Local" if not IS_ONLINE else "ai_council/Vectors/Online"