from backend.utils import compute_merton_pi
import numpy as np # type: ignore

def produce_merton_and_seed(mu: np.ndarray, Sigma: np.ndarray, r_f: float, A: float):
    pi = compute_merton_pi(mu, Sigma, r_f, A)
    return pi
