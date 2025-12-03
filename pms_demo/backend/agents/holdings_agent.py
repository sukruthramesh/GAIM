import pandas as pd
from ..utils import holdings_to_weights
from typing import Tuple

# holdings CSV expected schema provided below
def normalize_holdings(df: pd.DataFrame) -> Tuple[dict, float]:
    # assume df has columns: ticker, asset_class, market_value
    weights, total = holdings_to_weights(df, asset_class_col="asset_class", value_col="market_value")
    return weights, float(total)

def compute_pv_human_capital(profile: dict) -> float:
    # Simple PV heuristic: annual income * multiplier depending on age; replace with better method if needed
    age = profile.get("age", 35)
    income = profile.get("annual_income", 0)
    if age < 30:
        mult = 10
    elif age < 40:
        mult = 8
    elif age < 50:
        mult = 6
    else:
        mult = 4
    return income * mult
