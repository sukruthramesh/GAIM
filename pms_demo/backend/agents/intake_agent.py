# backend/agents/intake_agent.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd

class IntakeResult(BaseModel):
    client_profile: Dict[str, Any]
    holdings: Optional[List[Dict[str, Any]]] = None   # list of row dicts (serializable)
    liabilities: Optional[List[Dict[str, Any]]] = None
    docs: List[str] = []

def parse_holdings_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def parse_liabilities_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def intake_process(form_json: dict, holdings_csv_path: str = None, liabilities_csv_path: str = None, docs: list = None) -> IntakeResult:
    """
    Returns IntakeResult with holdings & liabilities serialized as list-of-dicts.
    Use pd.DataFrame internally if you need to run DataFrame operations.
    """
    holdings_df = parse_holdings_csv(holdings_csv_path) if holdings_csv_path else None
    liabilities_df = parse_liabilities_csv(liabilities_csv_path) if liabilities_csv_path else None

    holdings_serial = holdings_df.to_dict(orient="records") if holdings_df is not None else None
    liabilities_serial = liabilities_df.to_dict(orient="records") if liabilities_df is not None else None

    result = IntakeResult(
        client_profile=form_json,
        holdings=holdings_serial,
        liabilities=liabilities_serial,
        docs=docs or []
    )
    return result
