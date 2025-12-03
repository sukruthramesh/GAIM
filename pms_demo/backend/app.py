from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from typing import Optional
import pandas as pd
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

app = FastAPI(title="PMS Demo Orchestrator")

# --- helper to save uploaded file to disk
def save_upload(file: UploadFile, dest_folder="data/uploads"):
    os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, file.filename)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return dest

@app.post("/run_demo/")
async def run_demo(
    form_json: str = Form(...),  # JSON string of client form including Q1..Q12 answers and demographics
    holdings_csv: UploadFile = File(...),
    liabilities_csv: UploadFile = File(...),
    historical_prices_csv: UploadFile = File(...),
    docs_zip: Optional[UploadFile] = None
):
    # 1. Intake
    form = json.loads(form_json)
    holdings_path = save_upload(holdings_csv, dest_folder="data/uploads")
    liabilities_path = save_upload(liabilities_csv, dest_folder="data/uploads")
    hist_path = save_upload(historical_prices_csv, dest_folder="data/uploads")

    intake = intake_process(form, holdings_csv_path=holdings_path, liabilities_csv_path=liabilities_path, docs=[])
    holdings_df = intake.holdings_df
    liabilities_df = intake.liabilities_df

    # 2. holdings normalization
    current_weights_dict, total_value = normalize_holdings(holdings_df)
    pv_hc = compute_pv_human_capital(form)

    # 3. risk profiler
    qlist = [form.get(f"q{i}",0) for i in range(1,11)]
    q11_text = form.get("q11_text", None)
    q12_self = form.get("q12_self", None)
    include_optionals = form.get("include_optionals", False)
    A, meta = compute_risk_coefficient(qlist, q11_text, q12_self, include_optionals)

    # 4. RAG ingest & query (for demo, ingest folder 'data/docs')
    ingest_documents("data/docs", collection_name="pms_docs")
    rag_out = rag_query_summarize("market outlook for asset classes", top_k=5, collection_name="pms_docs")

    # 5. Estimate mu, Sigma from prices
    prices_df = pd.read_csv(hist_path, parse_dates=["Date"], index_col="Date")
    asset_order = list(prices_df.columns)
    mu_hist, Sigma = estimate_mu_sigma_from_prices(prices_df, freq="monthly")
    mu_final = blend_mu(mu_hist, rag_out["mu"], asset_order, w_hist=0.6, w_rag=0.4)
    r_f = 0.04  # placeholder; could be from RAG or input

    # 6. Merton & Markowitz
    pi = produce_merton_and_seed(mu_final, Sigma, r_f, A)
    # map pi to feasible positive weights (simple projection on [0,1] and renormalize)
    pi_pos = np.clip(pi, 0, None)
    if pi_pos.sum() <= 0:
        seed = np.ones(len(pi_pos))/len(pi_pos)
    else:
        seed = pi_pos / pi_pos.sum()

    # use current_weights vector in same asset_order
    current_weights_vec = np.array([current_weights_dict.get(a, 0.0) for a in asset_order])

    opt_res = optimise_portfolio(mu_final, Sigma, A, current_weights=current_weights_vec,
                                 lower_bounds=np.zeros(len(mu_final)), upper_bounds=np.ones(len(mu_final)),
                                 turnover_penalty=0.001)

    weights = opt_res["weights"]
    target_weights = {asset_order[i]: float(weights[i]) for i in range(len(weights))}
    # 7. explain
    explain_payload = {
        "client_profile": form,
        "A": A,
        "meta": meta,
        "target_weights": target_weights,
        "merton_seed": {asset_order[i]: float(seed[i]) for i in range(len(seed))}
    }
    explanation_text = explain_allocation(explain_payload)

    # 8. compliance
    compliance_rules = {"max_single_asset": 0.6, "required_cash_buffer": 0.03}
    compliance_res = check_compliance(target_weights, compliance_rules)

    # 9. trades
    trade_df = generate_trade_instructions(current_weights_dict, target_weights, total_value)
    trade_csv = "data/outputs/trades.csv"
    os.makedirs("data/outputs", exist_ok=True)
    trade_df.to_csv(trade_csv, index=False)

    return {
        "A": A,
        "meta": meta,
        "merton_seed": {asset_order[i]: float(seed[i]) for i in range(len(seed))},
        "target_weights": target_weights,
        "explanation": explanation_text,
        "compliance": compliance_res,
        "trades_csv": trade_csv
    }
