import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

DEFAULT_TICKER_MAP = {
    "SPY": "intl_equity",
    "VTI": "intl_equity",
    "GLD": "gold",
    "GOLDBEES": "gold",
    "IBOND": "domestic_bonds",
}

def load_ticker_map(path: str = None) -> Dict[str,str]:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        return dict(zip(df['ticker'].str.upper(), df['asset_class']))
    else:
        return {k.upper(): v for k, v in DEFAULT_TICKER_MAP.items()}

def infer_asset_class(row, ticker_map):
    t = str(row.get('ticker','')).upper()
    if t in ticker_map:
        return ticker_map[t]
    itype = str(row.get('instrument_type','')).lower()
    name = str(row.get('instrument_name','')).lower()
    if 'gold' in name or 'gold' in t:
        return 'gold'
    if 'bond' in itype or 'bond' in name:
        return 'domestic_bonds'
    if 'ulip' in name or 'ulip' in itype:
        return 'alternatives'
    if 'annuity' in name or 'annuity' in itype:
        return 'cash'
    if 'mf' in itype or 'fund' in name or 'mutual fund' in name:
        if 'debt' in name or 'bond' in name:
            return 'domestic_bonds'
        if 'gold' in name:
            return 'gold'
        return 'domestic_equity'
    return 'domestic_equity'

def compute_market_value(row):
    mv = row.get('market_value', None)
    if pd.notna(mv) and mv != '':
        try:
            return float(mv)
        except:
            pass
    q = row.get('quantity', None)
    p = row.get('price', None)
    if pd.notna(q) and pd.notna(p):
        try:
            return float(q) * float(p)
        except:
            pass
    return np.nan

def normalize_holdings_csv(path: str, ticker_map_path: str = None, base_currency: str = "INR") -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, dtype=str).fillna('')
    for col in ['quantity','price','market_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    ticker_map = load_ticker_map(ticker_map_path)
    df['market_value_computed'] = df.apply(compute_market_value, axis=1)
    df['needs_manual_valuation'] = df['market_value_computed'].isna()
    df['asset_class'] = df.apply(lambda r: infer_asset_class(r, ticker_map), axis=1)
    df['currency'] = df.get('currency', base_currency)
    df['market_value_final'] = df.apply(
        lambda r: r['market_value'] if pd.notna(r.get('market_value')) else r['market_value_computed'],
        axis=1
    )
    df['illiquid'] = df['instrument_type'].str.contains('private|illiquid|pe|private_equity', case=False, na=False)
    df['is_insurance'] = df['instrument_type'].str.contains('ulip|annuity|insurance', case=False, na=False)
    agg = df.groupby('asset_class')['market_value_final'].sum().to_dict()
    total = sum([v for v in agg.values() if not pd.isna(v)])
    weights = {k: (v/total if total>0 else 0.0) for k,v in agg.items()}
    cleaned_path = os.path.splitext(path)[0] + "_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    meta = {
        "total_value": total,
        "aggregation": agg,
        "weights": weights,
        "needs_manual_rows": df[df['needs_manual_valuation']].to_dict(orient='records')
    }
    return df, meta
