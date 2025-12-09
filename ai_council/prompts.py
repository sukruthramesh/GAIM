from pydantic import BaseModel
from typing import List, Optional, Dict
from ai_council.constants import MODELS


##### Scoring Output Format
class Scores(BaseModel):
    accuracy : int
    completeness : int
    grounding : int
    reasoning : int
    clarity : int

class scoring_output(BaseModel):
    scores : Scores
    confidence_estimate : float
    justification : str


##### Audit Output Format
class flag(BaseModel):
    scorer_id : str
    issue : str
    severity : str
class Audit_Report(BaseModel):
    audit_id : str
    flags : List[flag]
    drops : List[str]
    explanation : str
    normalization : Dict[str, float]


scoring_template ="""
    SYSTEM: You are an impartial evaluator that scores candidate answers to a user prompt. Use the rubric provided and be objective. 
    Return only the JSON object described below and nothing else.

    USER: Here is the ORIGINAL USER PROMPT:
    {user_prompt}

    Here is the CANDIDATE RESPONSE you must evaluate:
    {candidate_response}

    RUBRIC (score each 1-5; 5 = best):
    - accuracy: Is the content factually correct given known, verifiable facts? (1 = many factual errors or hallucinations; 5 = fully accurate)
    - completeness: Does it address all parts of the prompt? (1 = misses core parts; 5 = full coverage)
    - grounding: Does the response cite or reference verifiable sources or show evidence/reasoning that can be checked? (1 = unsupported claims; 5 = well-grounded)
    - reasoning: Are the logical steps coherent and correct? (1 = flawed reasoning; 5 = sound stepwise logic)
    - clarity: Is it readable, appropriately toned, and well-structured? (1 = confusing; 5 = clear & concise)

    Also provide a one-sentence justification for the total score and a confidence estimate between 0 and 1.

    Output Format : 
    {output_format}

    Notes:
    - Score numerically and be conservative: penalize minor hallucinations or unsupported numeric claims.
    - Do not refer to model names, internals, or policies in your justification.Constraints:
    - Output must be strictly valid JSON (use "null" for missing, numbers must be numeric).
    - Do not include trailing commas.
    - Do not include comments or explanatory text.
"""

expert_generation_template = """
    SYSTEM:
    You are a retrieval-grounded assistant. Use only the information in the CONTEXT. 
    If the answer is not in the context, say: "Not in context." 
    Do not guess or invent facts.

    FORMAT:
    1. Final answer (1-2 lines)
    2. Brief reasoning (1-2 lines)
    3. Snippets used (# or "none")

    USER:
    {user_prompt}

    CONTEXT:
    {context}

    RULES:
    - Base all statements strictly on the context.
    - Cite snippet numbers when used.
    - Keep responses short and precise.
"""


auditor_prompt_template = """
    SYSTEM: You are an independent auditor whose job is to inspect a scoring matrix produced by peer models and detect bias, collusion, or anomalous scoring patterns. Return only the JSON described below.

    USER: We provide:
    1) original_prompt: {user_prompt}
    2) responses: a JSON list of response objects:
    {responses}
    3) scoring_matrix: a JSON object where keys are scorer_ids and values are dictionaries mapping response_id -> score_obj
    e.g., {scoring_matrix}

    Task:
    1) Inspect scoring patterns for the following anomalies:
    - Self-scoring or allowed self-favoring (scorer giving systematically higher scores to a single partner)
    - Collusion: two or more scorers consistently upvoting each other across many prompts (pattern detection)
    - Extreme scorers: scorer that always gives very high (>=4.5) or very low (<=1.5) totals while variance is near zero
    - Outliers: scorer scores that deviate > 2 std from the mean for a response

    2) For each detected anomaly produce a corrective action:
    - normalization factor for that scorer (multiply all their scores by that factor)
    - or flag for human review (if severe)
    - or drop scorer from aggregation for this prompt

    Return JSON **only**:
    {output_format}
"""
