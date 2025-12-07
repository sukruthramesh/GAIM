from fastapi import FastAPI # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
from agents.intake_agent import intake_process
from agents.holdings_agent import normalize_holdings, compute_pv_human_capital
from agents.risk_profiler_agent import compute_risk_coefficient
from agents.market_rag_agent import ingest_documents, rag_query_summarize
from agents.estimator_agent import estimate_mu_sigma_from_prices, blend_mu
from agents.scoring_agent import produce_merton_and_seed
from agents.optimiser_agent import optimise_portfolio
from agents.explainability_agent import explain_allocation
from agents.compliance_agent import check_compliance
from agents.trade_generator_agent import generate_trade_instructions
from agents.utils_holdings import normalize_holdings_csv # type: ignore

app = FastAPI(title="PMS Demo Orchestrator - Local Run")

DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "outputs"), exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

RESPONSE_JSON_PATH = os.path.join(DATA_DIR, "response.json")
HOLDINGS_PATH = os.path.join(DATA_DIR, "holdings_sample.csv")
LIABILITIES_PATH = os.path.join(DATA_DIR, "liabilities_sample.csv")
HIST_PRICES_PATH = os.path.join(DATA_DIR, "historical_prices.csv")
DOCS_FOLDER = os.path.join(DATA_DIR, "docs")

@app.get("/run_demo_local/")
def run_demo_local():
    # 1. Read questionnaire responses from data/response.json
    if not os.path.exists(RESPONSE_JSON_PATH):
        return {"error":"Place response.json in data/ and re-run. See README_RUN.md for expected format."}
    with open(RESPONSE_JSON_PATH, "r", encoding="utf-8") as f:
        form = json.load(f)

    # 2. Normalize holdings CSV (if present)
    if os.path.exists(HOLDINGS_PATH):
        cleaned_df, meta = normalize_holdings_csv(HOLDINGS_PATH)
        holdings_df = cleaned_df
    else:
        return {"error":"Place holdings_sample.csv in data/ and re-run."}

    # 3. Intake -> build profile & holdings etc.
    intake = intake_process(form, holdings_csv_path=HOLDINGS_PATH, liabilities_csv_path=LIABILITIES_PATH, docs=[])
    current_weights_dict, total_value = normalize_holdings(holdings_df)
    pv_hc = compute_pv_human_capital(form)

    # 4. Risk profiler
    qlist = [int(form.get(f"q{i}",0)) for i in range(1,11)]
    q11_text = form.get("q11_text", None)
    q12_self = int(form.get("q12_self")) if form.get("q12_self") is not None else None
    include_optionals = bool(form.get("include_optionals", False))
    A, meta_risk = compute_risk_coefficient(qlist, q11_text, q12_self, include_optionals)

    # 5. RAG ingest & query
    ingest_documents(DOCS_FOLDER, collection_name="pms_docs")
    rag_out = rag_query_summarize("market outlook for asset classes", top_k=5, collection_name="pms_docs")

    # 6. Prices -> mu, Sigma
    if os.path.exists(HIST_PRICES_PATH):
        prices_df = pd.read_csv(HIST_PRICES_PATH, parse_dates=["Date"], index_col="Date")
        asset_order = list(prices_df.columns)
        mu_hist, Sigma = estimate_mu_sigma_from_prices(prices_df, freq="monthly")
        mu_final = blend_mu(mu_hist, rag_out["mu"], asset_order, w_hist=0.6, w_rag=0.4)
    else:
        return {"error":"Place historical_prices.csv in data/ and re-run."}

    r_f = 0.04

    # 7. Merton & Markowitz
    pi = produce_merton_and_seed(mu_final, Sigma, r_f, A)
    pi_pos = np.clip(pi, 0, None)
    if pi_pos.sum() <= 0:
        seed = np.ones(len(pi_pos))/len(pi_pos)
    else:
        seed = pi_pos / pi_pos.sum()

    current_weights_vec = np.array([current_weights_dict.get(a, 0.0) for a in asset_order])
    opt_res = optimise_portfolio(mu_final, Sigma, A, current_weights=current_weights_vec,
                                 lower_bounds=np.zeros(len(mu_final)), upper_bounds=np.ones(len(mu_final)),
                                 turnover_penalty=0.001)
    weights = opt_res["weights"]
    target_weights = {asset_order[i]: float(weights[i]) for i in range(len(weights))}

    explanation_payload = {
        "client_profile": form,
        "A": A,
        "meta": meta_risk,
        "target_weights": target_weights,
        "merton_seed": {asset_order[i]: float(seed[i]) for i in range(len(seed))}
    }
    explanation_text = explain_allocation(explanation_payload)

    compliance_rules = {"max_single_asset": 0.6, "required_cash_buffer": 0.03}
    compliance_res = check_compliance(target_weights, compliance_rules)

    trade_df = generate_trade_instructions(current_weights_dict, target_weights, total_value)
    trade_csv = os.path.join(DATA_DIR, "outputs", "trades.csv")
    trade_df.to_csv(trade_csv, index=False)

    return {
        "A": A,
        "meta": meta_risk,
        "merton_seed": {asset_order[i]: float(seed[i]) for i in range(len(seed))},
        "target_weights": target_weights,
        "explanation": explanation_text,
        "compliance": compliance_res,
        "trades_csv": trade_csv
    }
