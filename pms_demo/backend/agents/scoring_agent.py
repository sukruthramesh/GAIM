import numpy as np
from ..utils import compute_merton_pi

def produce_merton_and_seed(mu: np.ndarray, Sigma: np.ndarray, r_f: float, A: float):
    pi = compute_merton_pi(mu, Sigma, r_f, A)
    # map to feasible weights (no shorting in base)
    return pi
