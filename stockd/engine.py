from __future__ import annotations

import json
from datetime import date as dt_date, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stockd import settings
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.online_model import load_state, predict
from stockd.mentor import propose_news_deltas


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _risk_regime_shrink(macro: Dict) -> float:
    """
    Proxy psihologie/regim:
    VIX mare + DXY în creștere + SPX negativ => shrink.
    """
    try:
        vix = macro.get("series", {}).get("VIX", {}).get("last", None)
        spx = macro.get("series", {}).get("SPX", {}).get("ret_5d_pct", 0.0) or 0.0
        dxy = macro.get("series", {}).get("DXY", {}).get("ret_5d_pct", 0.0) or 0.0

        if vix is None:
            return 1.0

        vix = float(vix)
        if vix >= 25 and spx < 0:
            return 0.70
        if vix >= 22 and (spx < 0 or dxy > 0):
            return 0.80
        if vix <= 16:
            return 1.00
        return 0.90
    except Exception:
        return 0.90


def _fallback_heuristic(features: pd.DataFrame) -> np.ndarray:
    """
    Dacă nu există model_state, produc o predicție numerică simplă:
    - momentum (20d, 60d)
    - penalizare vol și drawdown
    """
    ret20 = pd.to_numeric(features.get("ret_20d"), errors="coerce").fillna(0.0).values
    ret60 = pd.to_numeric(features.get("ret_60d"), errors="coerce").fillna(0.0).values
    vol20 = pd.to_numeric(features.get("vol_20d"), errors="coerce").fillna(0.0).values
    mdd60 = pd.to_numeric(features.get("max_dd_60d"), errors="coerce").fillna(0.0).values  # negativ

    # scale conservator la săptămână
    base = 0.10 * ret20 + 0.05 * ret60 - 0.03 * vol20 + 0.02 * mdd60
    return np.clip(base, -5.0, 5.0)


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: dt_date,
    horizon_days: int,
) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    h = holdings.copy()
    h["Ticker"] = h["Ticker"].astype(str).str.upper().str.strip()
    h["Region"] = h["Region"].astype(str).str.upper().str.strip()
    h = h[["Ticker", "Region"]].drop_duplicates()

    # features
    feat_df, feat_debug = compute_ticker_features(prices_history, h)
    if feat_df.empty:
        out = h.copy()
        out["ER_Pct"] = 0.0
        return out

    # macro snapshot
    macro = get_macro_snapshot(datetime.combine(as_of, datetime.min.time())) if settings.ENABLE_MACRO else {"as_of": str(as_of), "series": {}, "regions": {}}
    shrink = _risk_regime_shrink(macro)

    # create model input frame (same columns as training)
    model_in = feat_df.copy()

    def s(name, key):
        v = macro.get("series", {}).get(name, {}).get(key, None)
        return float(v) if v is not None else 0.0

    # map macros
    model_in["vix_level"] = s("VIX", "last")
    model_in["dxy_ret_5d"] = s("DXY", "ret_5d_pct")
    model_in["oil_ret_5d"] = s("OIL", "ret_5d_pct")
    model_in["gold_ret_5d"] = s("GOLD", "ret_5d_pct")

    # region bench returns
    us_bench = macro.get("regions", {}).get("US", {}).get("bench_ret_5d_pct", 0.0) or 0.0
    eu_bench = macro.get("regions", {}).get("EU", {}).get("bench_ret_5d_pct", 0.0) or 0.0

    model_in["bench_ret_5d"] = 0.0
    model_in.loc[model_in["Region"] == "US", "bench_ret_5d"] = float(us_bench)
    model_in.loc[model_in["Region"] == "EU", "bench_ret_5d"] = float(eu_bench)
    model_in.loc[model_in["Region"] == "RO", "bench_ret_5d"] = pd.to_numeric(model_in["ro_proxy_ret_5d"], errors="coerce").fillna(0.0)

    # numeric prediction
    state = load_state(settings.MODEL_STATE_JSON)
    if state and state.get("ok"):
        base_pred = predict(model_in, state)
    else:
        base_pred = _fallback_heuristic(model_in)

    # apply regime shrink
    base_pred = np.clip(base_pred * float(shrink), -settings.MAX_ABS_ER_PCT, settings.MAX_ABS_ER_PCT)

    # news delta from LLM (batch, capped)
    deltas = { (r["Ticker"], r["Region"]): 0.0 for _, r in model_in.iterrows() }
    reasons = { (r["Ticker"], r["Region"]): "" for _, r in model_in.iterrows() }

    if settings.OPENAI_API_KEY and settings.ENABLE_LLM_NEWS_ADJ:
        items = []
        for i, r in model_in.iterrows():
            items.append({"Ticker": r["Ticker"], "Region": r["Region"], "base_er_pct": float(base_pred[i])})
        out = propose_news_deltas(items, macro_snapshot=macro)
        if isinstance(out, dict) and out.get("ok") and isinstance(out.get("deltas"), list):
            for d in out["deltas"]:
                k = (str(d.get("Ticker","")).upper().strip(), str(d.get("Region","")).upper().strip())
                delta_pp = float(d.get("delta_pp", 0.0) or 0.0)
                delta_pp = _clip(delta_pp, -settings.MAX_NEWS_DELTA_PP, settings.MAX_NEWS_DELTA_PP)
                if bool(d.get("ambiguous", False)):
                    delta_pp = 0.0
                if k in deltas:
                    deltas[k] = delta_pp
                    reasons[k] = str(d.get("reason",""))[:120]

    # final
    er = []
    for i, r in model_in.iterrows():
        k = (r["Ticker"], r["Region"])
        final = float(base_pred[i]) + float(deltas.get(k, 0.0))
        final = _clip(final, -settings.MAX_ABS_ER_PCT, settings.MAX_ABS_ER_PCT)
        er.append(final)

    out_df = model_in[["Ticker","Region"]].copy()
    out_df["ER_Pct"] = er
    return out_df
