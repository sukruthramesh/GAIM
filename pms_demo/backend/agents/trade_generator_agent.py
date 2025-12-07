import pandas as pd # type: ignore

def generate_trade_instructions(current_weights: dict, target_weights: dict, total_portfolio_value: float, min_lot_size: dict=None):
    trades = []
    for k in target_weights.keys():
        current_w = current_weights.get(k, 0.0)
        target_w = target_weights[k]
        delta_w = target_w - current_w
        delta_amt = delta_w * total_portfolio_value
        if abs(delta_amt) < 1e-6:
            continue
        if delta_amt > 0:
            trades.append({"asset_class": k, "action": "BUY", "amount": round(delta_amt,2)})
        else:
            trades.append({"asset_class": k, "action": "SELL", "amount": round(-delta_amt,2)})
    df = pd.DataFrame(trades)
    return df
