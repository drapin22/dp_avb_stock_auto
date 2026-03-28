from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from stockd import settings

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "max_dd_60d",
    "beta",
    "volume_ratio",
    "peer_rel_20d",
    "regime",
    "vix_level",
    "dxy_ret_5d",
    "bench_ret_5d",
]

MIN_TRAIN_WEEKS = 8
RIDGE_ALPHA = 0.5


def compute_features_as_of(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Compute per-ticker features using only data up to as_of."""
    p = prices[prices["Date"] <= as_of].copy()
    if p.empty:
        return pd.DataFrame(columns=["Ticker", "Region"] + FEATURE_COLS)

    proxy = (
        p.groupby(["Region", "Date"])["Close"]
        .mean().pct_change().rename("ProxyRet").reset_index()
    )

    reg_ret20 = {}
    for reg, g in p.groupby("Region"):
        rets = []
        for _, tg in g.groupby("Ticker"):
            c = tg.sort_values("Date")["Close"].reset_index(drop=True)
            if len(c) >= 21:
                rets.append((float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100.0)
        reg_ret20[reg] = float(np.nanmean(rets)) if rets else 0.0

    has_vol = "Volume" in p.columns
    rows = []

    for (tkr, reg), g in p.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        close = g["Close"].astype(float).reset_index(drop=True)
        n = len(close)

        def sr(lb):
            return (float(close.iloc[-1]) / float(close.iloc[-lb-1]) - 1.0) * 100.0 if n >= lb+1 else 0.0

        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std() * np.sqrt(252) * 100.0) if len(rets) >= 10 else 20.0
        window = close.tail(60) if n >= 10 else close
        max_dd = float(((window / window.cummax()) - 1.0).min() * 100.0)

        prx = proxy[proxy["Region"] == reg].set_index("Date")["ProxyRet"]
        ar = rets.copy()
        ar.index = g["Date"].iloc[1:len(rets)+1].values
        ar = ar.reindex(prx.index).dropna()
        if len(ar) >= 10:
            vb = float(prx.reindex(ar.index).var())
            beta = float(ar.cov(prx.reindex(ar.index)) / vb) if vb > 1e-12 else 1.0
        else:
            beta = 1.0

        if has_vol:
            vc = pd.to_numeric(g["Volume"], errors="coerce").astype(float).reset_index(drop=True)
            avg20 = float(vc.tail(20).mean())
            volume_ratio = float(vc.iloc[-1] / avg20) if avg20 > 0 else 1.0
        else:
            volume_ratio = 1.0

        ret_20d = sr(20)
        peer_rel = ret_20d - reg_ret20.get(reg, 0.0)
        sma60 = float(close.tail(60).mean()) if n >= 20 else float(close.mean())
        regime = 1.0 if float(close.iloc[-1]) > sma60 else 0.0

        rows.append({
            "Ticker": tkr, "Region": reg,
            "ret_5d": sr(5), "ret_20d": ret_20d, "ret_60d": sr(60),
            "vol_20d": vol_20d, "max_dd_60d": max_dd, "beta": beta,
            "volume_ratio": volume_ratio, "peer_rel_20d": peer_rel, "regime": regime,
            "vix_level": 0.0, "dxy_ret_5d": 0.0, "bench_ret_5d": 0.0,
        })

    return pd.DataFrame(rows)


def compute_forward_return(prices: pd.DataFrame, as_of: pd.Timestamp, horizon: int = 5) -> pd.DataFrame:
    rows = []
    for (tkr, reg), g in prices.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        past = g[g["Date"] <= as_of]
        future = g[g["Date"] > as_of].head(horizon)
        if past.empty or future.empty:
            continue
        p0 = float(past.iloc[-1]["Close"])
        p1 = float(future.iloc[-1]["Close"])
        if p0 > 0:
            rows.append({"Ticker": tkr, "Region": reg, "fwd_ret": (p1/p0-1.0)*100.0, "AsOf": as_of})
    return pd.DataFrame(rows)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    n, d = X.shape
    A = X.T @ X + alpha * np.eye(d)
    try:
        coef = np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.zeros(d)
    intercept = float(np.mean(y) - X.mean(axis=0) @ coef)
    return coef, intercept


def run_walkforward(prices: pd.DataFrame, horizon: int = 5):
    prices["Date"] = pd.to_datetime(prices["Date"])
    mondays = sorted(set(d for d in prices["Date"].unique() if pd.Timestamp(d).dayofweek == 0))

    train_feats = []
    eval_rows = []
    current_coef = np.zeros(len(FEATURE_COLS))
    current_intercept = 0.15

    for i, monday in enumerate(mondays[:-1]):
        ts = pd.Timestamp(monday)
        feats = compute_features_as_of(prices, ts)
        fwds = compute_forward_return(prices, ts, horizon)
        merged = feats.merge(fwds, on=["Ticker","Region"], how="left")
        merged["AsOf"] = ts

        if i >= MIN_TRAIN_WEEKS and train_feats:
            train_df = pd.concat(train_feats, ignore_index=True).dropna(subset=["fwd_ret"] + FEATURE_COLS)
            if len(train_df) >= 4:
                X_tr = train_df[FEATURE_COLS].fillna(0.0).values
                y_tr = train_df["fwd_ret"].values
                current_coef, current_intercept = ridge_fit(X_tr, y_tr)

            X_pred = merged[FEATURE_COLS].fillna(0.0).values
            preds = current_intercept + X_pred @ current_coef

            for j, (_, row) in enumerate(merged.iterrows()):
                predicted = float(preds[j])
                actual = float(row["fwd_ret"]) if not pd.isna(row.get("fwd_ret")) else float("nan")
                err = (predicted - actual) if not np.isnan(actual) else float("nan")
                eval_rows.append({
                    "WeekStart": str(ts.date()),
                    "Ticker": row["Ticker"], "Region": row["Region"],
                    "Predicted_ER": round(predicted, 4),
                    "Actual_ER": round(actual, 4) if not np.isnan(actual) else None,
                    "Error_Pct": round(err, 4) if not np.isnan(err) else None,
                    "AbsError_Pct": round(abs(err), 4) if not np.isnan(err) else None,
                    "DirectionHit": 1.0 if (predicted * actual > 0) else (0.0 if not np.isnan(actual) else None),
                })

        train_feats.append(merged)

    all_df = pd.concat(train_feats, ignore_index=True).dropna(subset=["fwd_ret"] + FEATURE_COLS) if train_feats else pd.DataFrame()
    if len(all_df) >= 4:
        X_all = all_df[FEATURE_COLS].fillna(0.0).values
        current_coef, current_intercept = ridge_fit(X_all, all_df["fwd_ret"].values)

    last = mondays[-2] if len(mondays) >= 2 else mondays[-1]
    state = {
        "feature_cols": FEATURE_COLS,
        "coef": current_coef.tolist(),
        "intercept": float(current_intercept),
        "trained_on_target_date": str(pd.Timestamp(last).date()),
        "n_samples": int(len(all_df)),
    }
    return pd.DataFrame(eval_rows), state


def retrain_and_save(prices_path=None):
    from stockd.scoring import compute_scores, save_scores

    prices_path = prices_path or settings.PRICES_HISTORY
    if not prices_path.exists():
        return {"error": f"Prices not found: {prices_path}"}

    prices = pd.read_csv(prices_path)
    prices["Date"]  = pd.to_datetime(prices["Date"],  errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"],  errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["Region"] = prices["Region"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])

    eval_df, state = run_walkforward(prices)

    model_path = settings.DATA_DIR / "model_state.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    summary = {"n_samples": state.get("n_samples", 0), "n_eval_rows": 0}
    if not eval_df.empty:
        eval_path = settings.DATA_DIR / "backtest_eval.csv"
        eval_df.to_csv(eval_path, index=False)
        summary["n_eval_rows"] = len(eval_df)
        scores = compute_scores(eval_df)
        if not scores.empty:
            save_scores(scores)
            summary["n_scored"] = len(scores)

    coef = state.get("coef", [])
    names = state.get("feature_cols", [])
    importance = sorted(zip(names, coef), key=lambda x: abs(x[1]), reverse=True)
    summary["top_features"] = {k: round(v, 4) for k, v in importance[:5]}
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    result = retrain_and_save()
    print(json.dumps(result, indent=2))
