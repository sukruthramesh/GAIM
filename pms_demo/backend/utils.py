import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import List, Optional, Tuple
import cvxpy as cp # type: ignore

# ---------------------------
# Risk scoring utilities
# ---------------------------

def map_q12_continuous(s: int) -> float:
    # continuous mapping 1..10 -> 0..4
    val = 4.0 * (10 - s) / 9.0
    return max(0.0, min(4.0, val))

def compute_A(q_mandatory: List[int],
              q11: Optional[float] = None,
              q12_s: Optional[int] = None,
              include_optionals: bool = True,
              A_min: float = 1.0,
              alpha: float = 6.0,
              beta: float = 2.0,
              kappa: float = 4.0,
              A_max: float = 15.0) -> Tuple[float, dict]:
    """
    Computes risk aversion coefficient A using the hybrid linear+log formula.
    q_mandatory: list of 10 ints (0..4)
    q11: optional semantic score (0..4)
    q12_s: self-rated 1..10 (optional)
    include_optionals: whether to include Q11/Q12 when present
    returns (A, meta)
    """
    if len(q_mandatory) != 10:
        raise ValueError("q_mandatory must be length 10")

    RAS_mand = sum(q_mandatory)
    RAS_total = RAS_mand
    n_opt = 0
    q11_val = None
    q12_val = None
    if include_optionals and (q11 is not None):
        q11_val = max(0.0, min(4.0, float(q11)))
        RAS_total += q11_val
        n_opt += 1
    if include_optionals and (q12_s is not None):
        q12_val = map_q12_continuous(q12_s)
        RAS_total += q12_val
        n_opt += 1

    RAS_scale = 40 + 4 * n_opt
    x = RAS_total / RAS_scale
    A = A_min + alpha * x + beta * math.log(1 + kappa * x)
    A = max(A_min, min(A, A_max))
    meta = {
        "RAS_mand": RAS_mand,
        "q11": q11_val,
        "q12_mapped": q12_val,
        "n_opt": n_opt,
        "RAS_total": RAS_total,
        "RAS_scale": RAS_scale,
        "x": x
    }
    return A, meta

# ---------------------------
# Merton formula
# ---------------------------

def compute_merton_pi(mu: np.ndarray, Sigma: np.ndarray, r_f: float, A: float) -> np.ndarray:
    """
    mu: expected returns (vector) - can be absolute or excess? We'll use absolute returns and subtract r_f below.
    Sigma: covariance matrix
    """
    excess = mu - r_f
    inv = np.linalg.inv(Sigma)
    pi = (1.0 / A) * inv.dot(excess)
    return pi

# ---------------------------
# Markowitz solver (QP)
# ---------------------------

def markowitz_optimize(mu: np.ndarray, Sigma: np.ndarray, A: float, w0: Optional[np.ndarray] = None,
                       lower_bounds: Optional[np.ndarray] = None, upper_bounds: Optional[np.ndarray] = None,
                       l1_turnover_penalty: float = 0.0) -> dict:
    """
    Solve: maximize w^T mu - 0.5 * A * w^T Sigma w - tau * ||w - w0||_1
    s.t. sum(w)=1, lower_bounds <= w <= upper_bounds
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(w @ mu - 0.5 * A * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if lower_bounds is not None:
        constraints.append(w >= lower_bounds)
    if upper_bounds is not None:
        constraints.append(w <= upper_bounds)
    if l1_turnover_penalty > 0 and w0 is not None:
        objective = cp.Maximize(w @ mu - 0.5 * A * cp.quad_form(w, Sigma) - l1_turnover_penalty * cp.norm1(w - w0))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    if w.value is None:
        raise Exception("QP failed to find solution")
    w_opt = np.array(w.value).flatten()
    return {
        "weights": w_opt,
        "expected_return": float(w_opt.dot(mu)),
        "expected_variance": float(w_opt.T.dot(Sigma).dot(w_opt))
    }

# ---------------------------
# helper to compute current weights from holdings CSV
# ---------------------------

def holdings_to_weights(df_holdings: pd.DataFrame, asset_class_col: str = "asset_class", value_col: str = "market_value"):
    """
    Expects df_holdings with columns [asset_class, market_value].
    Returns dict asset_class -> weight
    """
    grouped = df_holdings.groupby(asset_class_col)[value_col].sum()
    total = grouped.sum()
    weights = (grouped / total).to_dict()
    return weights, total
