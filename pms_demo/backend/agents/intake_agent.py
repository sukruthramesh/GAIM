from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

# Simple intake parser - reads uploaded CSVs; UI will POST files

class IntakeResult(BaseModel):
    client_profile: Dict[str, Any]
    holdings_df: pd.DataFrame
    liabilities_df: pd.DataFrame
    docs: List[str]  # paths to doc files for RAG

def parse_holdings_csv(file_path: str):
    df = pd.read_csv(file_path)
    # expect columns validated externally (see schema)
    return df

def parse_liabilities_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df

def intake_process(form_json: dict, holdings_csv_path: str = None, liabilities_csv_path: str = None, docs: list = None):
    # form_json contains questionnaire answers and client demographics
    holdings = parse_holdings_csv(holdings_csv_path) if holdings_csv_path else None
    liabilities = parse_liabilities_csv(liabilities_csv_path) if liabilities_csv_path else None
    result = IntakeResult(
        client_profile=form_json,
        holdings_df=holdings,
        liabilities_df=liabilities,
        docs=docs or []
    )
    return result
