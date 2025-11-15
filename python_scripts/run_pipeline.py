# run_pipeline.py
import os, json
from pipeline import run_pipeline

user_text = """I want to invest in a 5-year USD fixed corporate bond, 6% coupon, semi annual payments.
I am an INR-denominated fund in India with expensive USD borrowing.
Create a synthetic exposure that minimizes FX and borrowing cost."""

sample_market_inputs = {
    "spot": 83.0,
    "r_inr": 0.07,
    "r_usd": 0.04,
    "tenor_years": 5.0,
    # optional: "market_forward": 86.0,
    "txn_cost_p_a": 0.001,
    "notional_usd": 100.0,
    "coupon_rate": 0.06,
    "freq": 2
}

out = run_pipeline(user_text, sample_market_inputs)
print(json.dumps(out, indent=2))
