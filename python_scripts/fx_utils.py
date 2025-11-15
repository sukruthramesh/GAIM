# fx_utils.py
from typing import Optional, List, Dict
import math

def compute_forward_by_cip(spot: float, r_domestic: float, r_foreign: float, T_years: float) -> float:
    """
    Discrete CIP forward:
    F = S * ((1+r_foreign)**T) / ((1+r_domestic)**T)
    r_domestic and r_foreign are decimals (e.g., 0.07 for 7%)
    """
    F = spot * ((1 + r_foreign) ** T_years) / ((1 + r_domestic) ** T_years)
    return F

def annualized_forward_premium(spot: float, forward: float, T_years: float) -> float:
    return (forward / spot) ** (1 / T_years) - 1

def implied_synthetic_usd_funding_rate(r_domestic: float, annual_forward_premium: float, txn_cost_p_a: float=0.0) -> float:
    return r_domestic + annual_forward_premium + txn_cost_p_a

def decision_fx_swap(spot: float, r_domestic: float, r_usd_borrow: float, T_years: float,
                     market_forward: Optional[float] = None, txn_cost_p_a: float = 0.001) -> Dict:
    """
    Compares synthetic (spot+forward) implied USD funding with direct USD borrowing.
    """
    F = market_forward if market_forward is not None else compute_forward_by_cip(spot, r_domestic, r_usd_borrow, T_years)
    premium = annualized_forward_premium(spot, F, T_years)
    implied = implied_synthetic_usd_funding_rate(r_domestic, premium, txn_cost_p_a)
    delta = implied - r_usd_borrow
    decision = "Preferred" if delta < 0 else "Not Preferred"
    return {
        "spot": spot,
        "forward": F,
        "annual_forward_premium": premium,
        "implied_synthetic_usd_funding_rate": implied,
        "direct_usd_borrow_cost": r_usd_borrow,
        "delta": delta,
        "decision": decision
    }

def generate_cashflow_schedule(notional_usd: float, coupon_rate_annual: float, freq_per_year: int, tenor_years: float) -> List[Dict]:
    """
    Create list of dicts: {'period', 'time_years', 'type', 'amount_usd'}
    """
    periods = int(tenor_years * freq_per_year)
    coupon_per_period = notional_usd * (coupon_rate_annual / freq_per_year)
    schedule = []
    for i in range(1, periods + 1):
        t = i / freq_per_year
        schedule.append({"period": i, "time_years": t, "type": "coupon", "amount_usd": round(coupon_per_period, 6)})
    schedule[-1]["type"] = "coupon+principal"
    schedule[-1]["amount_usd"] = round(coupon_per_period + notional_usd, 6)
    return schedule

def convert_cashflows_to_inr(schedule: List[Dict], forward_curve: Dict[float, float]) -> List[Dict]:
    """
    forward_curve: mapping time_years -> forward_rate (INR per USD)
    """
    converted = []
    for item in schedule:
        t = item["time_years"]
        # find nearest forward (exact matches expected in our implementation)
        F = forward_curve.get(t)
        if F is None:
            # fallback: use longest available
            F = list(forward_curve.values())[-1]
        converted.append({
            "period": item["period"],
            "time_years": t,
            "usd_amount": item["amount_usd"],
            "fx_forward_used": F,
            "inr_amount": round(item["amount_usd"] * F, 6)
        })
    return converted
