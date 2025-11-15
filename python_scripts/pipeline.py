# pipeline.py
import os, json
import openai
from agents import Agent, PROBLEM_NORMALIZER_PROMPT, ENUMERATOR_PROMPT, EVALUATOR_PROMPT, SELECTOR_PROMPT, AUDITOR_PROMPT, VALIDATOR_PROMPT
from fx_utils import compute_forward_by_cip, decision_fx_swap, generate_cashflow_schedule, convert_cashflows_to_inr

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gpt-4o")

# instantiate agents (same model by default)
A1 = Agent("Normalizer", PROBLEM_NORMALIZER_PROMPT, MODEL)
A2 = Agent("Enumerator", ENUMERATOR_PROMPT, MODEL)
A3 = Agent("Evaluator", EVALUATOR_PROMPT, MODEL)
A4 = Agent("Selector", SELECTOR_PROMPT, MODEL)
A5 = Agent("Auditor", AUDITOR_PROMPT, MODEL)
A6 = Agent("Validator", VALIDATOR_PROMPT, MODEL)

def run_pipeline(user_text: str, sample_market_inputs: dict = None):
    # 1 Normalizer
    norm = A1.run(user_text)
    # 2 Enumerator
    enumerated = A2.run(norm)
    # 3 Evaluator
    eval_table = A3.run(enumerated)
    # 4 Selector
    selected = A4.run(eval_table + "\n\n" + norm)
    # 5 Auditor
    audit = A5.run(enumerated + "\n\n" + eval_table)
    # 6 Validator
    validation = A6.run(selected)

    # FX numeric check (spot+forward) if market inputs provided
    fx_result = None
    if sample_market_inputs:
        s = sample_market_inputs["spot"]
        r_inr = sample_market_inputs["r_inr"]
        r_usd_borrow = sample_market_inputs["r_usd"]
        T = sample_market_inputs.get("tenor_years", 5.0)
        market_forward = sample_market_inputs.get("market_forward")  # optional
        fx_result = decision_fx_swap(s, r_inr, r_usd_borrow, T, market_forward, txn_cost_p_a=sample_market_inputs.get("txn_cost_p_a", 0.001))

        # cashflow schedule + conversion example
        notional = sample_market_inputs.get("notional_usd", 100.0)
        coupon = sample_market_inputs.get("coupon_rate", 0.06)
        freq = sample_market_inputs.get("freq", 2)
        schedule = generate_cashflow_schedule(notional, coupon, freq, T)
        # create forward curve mapping each coupon time to forward F (assume same F for simplicity)
        forward_curve = {item["time_years"]: fx_result["forward"] for item in schedule}
        converted = convert_cashflows_to_inr(schedule, forward_curve)
    else:
        schedule = converted = None

    return {
        "normalized": norm,
        "enumerated": enumerated,
        "evaluation": eval_table,
        "selected": selected,
        "audit": audit,
        "validation": validation,
        "fx_numeric": fx_result,
        "schedule": schedule,
        "inr_converted_schedule": converted
    }
