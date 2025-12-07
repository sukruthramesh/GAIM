import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.covariance import LedoitWolf # type: ignore

def estimate_mu_sigma_from_prices(prices_df: pd.DataFrame, freq: str = "monthly"):
    returns = prices_df.pct_change().dropna()
    if freq == "monthly":
        scale = 12
    else:
        scale = 252
    mu_hist = returns.mean() * scale
    lw = LedoitWolf()
    lw.fit(returns.values)
    Sigma_shrink = lw.covariance_ * scale
    mu_arr = mu_hist.values
    return mu_arr, Sigma_shrink

def blend_mu(mu_hist: np.ndarray, mu_rag_dict: dict, asset_order: list, w_hist=0.6, w_rag=0.4):
    mu_rag_vec = np.array([mu_rag_dict.get(a, 0.02) for a in asset_order])
    mu_final = w_hist * mu_hist + w_rag * mu_rag_vec
    return mu_final
