import numpy as np # type: ignore
from backend.utils import markowitz_optimize

def optimise_portfolio(mu: np.ndarray, Sigma: np.ndarray, A: float, current_weights: np.ndarray = None,
                       lower_bounds=None, upper_bounds=None, turnover_penalty: float = 0.001):
    n = len(mu)
    if lower_bounds is None:
        lower_bounds = np.zeros(n)
    if upper_bounds is None:
        upper_bounds = np.ones(n)
    w0 = current_weights if current_weights is not None else np.ones(n) / n
    res = markowitz_optimize(mu, Sigma, A, w0=w0, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                             l1_turnover_penalty=turnover_penalty)
    return res
