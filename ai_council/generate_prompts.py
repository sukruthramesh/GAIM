def generate_scoring_prompt(user_prompt, candidate_response):
    prompt ="""
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

    Also provide a one-sentence justification for the total score.

    Return JSON only with these fields:
    {
    "scores": {
        "accuracy": int,
        "completeness": int,
        "grounding": int,
        "reasoning": int,
        "clarity": int
    },
    "confidence_estimate": float, # 0.0-1.0; your best estimate of how confident you are
    "justification": "one-sentence justification"
    }

    Notes:
    - Score numerically and be conservative: penalize minor hallucinations or unsupported numeric claims.
    - Do not refer to model names, internals, or policies in your justification.

    """
    prompt = prompt.replace('{user_prompt}',user_prompt)
    prompt = prompt.replace('{candidate_response}', candidate_response)
    return prompt

    # "total": float,             # weighted sum (weights defined below)
    # WEIGHTS (apply when computing "total"):
    # total = 0.35*accuracy + 0.25*completeness + 0.20*grounding + 0.15*reasoning + 0.05*clarity



def generate_auditor_prompt(user_prompt, responses, scoring_matrix):
    prompt = """SYSTEM: You are an independent auditor whose job is to inspect a scoring matrix produced by peer models and detect bias, collusion, or anomalous scoring patterns. Return only the JSON described below.

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
                {
                "audit_id": "auditor_run_<uuid>",
                "flags": [
                    {"scorer_id":"expert_2", "issue":"consistent mutual upvoting with expert_3", "severity":"medium", "action":"apply_penalty"},
                    ...
                ],
                "normalization": {
                    "expert_1": 1.0,
                    "expert_2": 0.7,
                    "expert_3": 1.0
                },
                "drops": ["scorer_id_to_drop_if_any"],
                "adjusted_scoring_matrix": { ... same shape as input but with adjusted 'total' values ... },
                "explanation": "one-paragraph summary of why adjustments were made"
                }"""

    return prompt.replace("{user_prompt}",user_prompt).replace('{responses}', responses).replace('{scoring_matrix}',scoring_matrix)
