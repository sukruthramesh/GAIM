# streamlit_app.py
import streamlit as st
import os, json
from pipeline import run_pipeline

st.title("Synthetic Cashflow Replicator — Demo")

st.markdown("Enter instrument description and market inputs.")

user_text = st.text_area("Instrument / Objective", value="I want to invest in a 5-year USD fixed corporate bond, 6% coupon, semiannual. INR fund; USD borrowing expensive; minimize cost.")

spot = st.number_input("Spot USD/INR", value=83.0)
r_inr = st.number_input("Your INR borrowing rate (annual %)", value=7.0) / 100.0
r_usd = st.number_input("Market USD borrowing rate (annual %)", value=4.0) / 100.0
market_forward = st.number_input("Market 5yr forward (optional, INR per USD) (0 to compute by CIP)", value=0.0)
txn_cost = st.number_input("Estimated txn cost p.a. (bps)", value=10.0) / 10000.0
notional = st.number_input("Notional USD", value=100.0)
coupon = st.number_input("Coupon rate (annual %)", value=6.0) / 100.0

if st.button("Run Pipeline"):
    market_inputs = {
        "spot": spot,
        "r_inr": r_inr,
        "r_usd": r_usd,
        "tenor_years": 5.0,
        "market_forward": None if market_forward == 0.0 else market_forward,
        "txn_cost_p_a": txn_cost,
        "notional_usd": notional,
        "coupon_rate": coupon,
        "freq": 2
    }
    with st.spinner("Calling pipeline..."):
        out = run_pipeline(user_text, market_inputs)
    st.subheader("Normalizer")
    st.text(out["normalized"])
    st.subheader("Top recommended structure (Selector)")
    st.text(out["selected"])
    st.subheader("FX numeric comparison (Spot+Forward)")
    st.json(out["fx_numeric"])
    st.subheader("Converted cashflow schedule (INR)")
    st.table(out["inr_converted_schedule"])
