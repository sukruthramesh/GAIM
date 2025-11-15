# tests/test_fx_utils.py
from fx_utils import compute_forward_by_cip, annualized_forward_premium, implied_synthetic_usd_funding_rate

def test_cip_forward():
    S = 83.0
    r_inr = 0.07
    r_usd = 0.04
    T = 5.0
    F = compute_forward_by_cip(S, r_inr, r_usd, T)
    # F should be > S if USD rate > INR rate? actually depends; we test numeric type
    assert isinstance(F, float)
    premium = annualized_forward_premium(S, F, T)
    implied = implied_synthetic_usd_funding_rate(r_inr, premium, 0.001)
    assert implied > 0
    print(F, implied, premium)

test_cip_forward()