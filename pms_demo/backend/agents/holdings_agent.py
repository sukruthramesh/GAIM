import pandas as pd # type: ignore
from utils import holdings_to_weights
from typing import Tuple

def normalize_holdings(df: pd.DataFrame) -> Tuple[dict, float]:
    weights, total = holdings_to_weights(df, asset_class_col="asset_class", value_col="market_value_final" if "market_value_final" in df.columns else "market_value")
    return weights, float(total)

def compute_pv_human_capital(profile: dict) -> float:
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
    try:
        return income * mult
    except:
        return 0.0
