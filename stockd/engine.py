from __future__ import annotations
from datetime import date
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from stockd import settings
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.online_model import load_state, predict
from stockd.mentor import propose_news_deltas
from stockd.calibration import load_calibration
from stockd.scoring import load_scores


def _safe_float(x, default=0.0):
    try:
        return default if pd.isna(x) else float(x)
    except Exception:
        return default


def _regime_shrink(er: float, macro: Dict, stock_regime: float = 1.0) -> Tuple[float, Dict]:
    vix = _safe_float(macro.get("vix_level"), 0.0)
    dxy = _safe_float(macro.get("dxy_ret_5d"), 0.0)
    shrink = 1.0
    # Macro risk-off
    if vix >= 28: shrink *= 0.70
    elif vix >= 22: shrink *= 0.85
    if dxy >= 1.5: shrink *= 0.80
    elif dxy >= 0.8: shrink *= 0.90
    # Stock-level regime: penalise BUY signals when price < 60d SMA
    if stock_regime < 0.5 and er > 0:
        shrink *= 0.75
    return er * shrink, {"shrink": round(shrink, 3), "vix": vix, "dxy": dxy}


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date | None = None,
    horizon_days: int = 5,
) -> pd.DataFrame:
    as_of = as_of or date.today()
    as_of_ts = pd.Timestamp(as_of)

    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    h = holdings.copy()
    h["Ticker"] = h["Ticker"].astype(str).str.upper().str.strip()
    h["Region"] = h["Region"].astype(str).str.upper().str.strip()
    h = h.drop_duplicates(subset=["Ticker", "Region"])

    feats = compute_ticker_features(prices_history, as_of=as_of_ts)
    model_in = h.merge(feats, on=["Ticker", "Region"], how="left")

    macro = get_macro_snapshot(as_of=as_of)

    # Map macro into model columns
    for src in ["vix_level", "dxy_ret_5d", "oil_ret_5d", "gold_ret_5d", "bench_ret_5d"]:
        model_in[src] = _safe_float(macro.get(src), 0.0)

    # RO benchmark fallback
    if "ro_proxy_ret_5d" in model_in.columns:
        mask = (model_in["Region"] == "RO") & (pd.to_numeric(model_in["bench_ret_5d"], errors="coerce").fillna(0.0) == 0.0)
        model_in.loc[mask, "bench_ret_5d"] = pd.to_numeric(model_in.loc[mask, "ro_proxy_ret_5d"], errors="coerce").fillna(0.0)

    # Fill new feature columns if missing (backward compat)
    new_cols = {"volume_ratio": 1.0, "peer_rel_20d": 0.0, "regime": 1.0, "ret_5d": 0.0}
    for col, default in new_cols.items():
        if col not in model_in.columns:
            model_in[col] = default
        else:
            model_in[col] = pd.to_numeric(model_in[col], errors="coerce").fillna(default)

    state = load_state()
    base_pred = predict(model_in, state=state)

    deltas = {}
    if settings.ENABLE_LLM_NEWS_ADJ and settings.OPENAI_API_KEY:
        try:
            deltas = propose_news_deltas(
                tickers=model_in[["Ticker","Region"]].to_dict(orient="records"),
                macro=macro, as_of=as_of,
            )
        except Exception:
            deltas = {}

    calib = load_calibration()
    calib_regions = (calib or {}).get("regions", {})
    scores = load_scores()
    scores_map = {}
    if scores is not None and not scores.empty:
        for _, r in scores.iterrows():
            scores_map[(str(r["Ticker"]).upper().strip(), str(r["Region"]).upper().strip())] = {
                "score": _safe_float(r.get("score_0_100"), 50.0),
                "tier":  str(r.get("confidence_tier") or "").strip(),
            }

    out_rows = []
    for i, r in model_in.iterrows():
        t, reg = r["Ticker"], r["Region"]
        raw    = _safe_float(base_pred[i], 0.0)
        delta  = _safe_float(deltas.get((t, reg), 0.0), 0.0)
        pre_cal = raw + delta

        rcfg = calib_regions.get(reg, {})
        mult = _safe_float(rcfg.get("mult"), 1.0)
        bias = _safe_float(rcfg.get("bias"), 0.0)
        clip = _safe_float(rcfg.get("clip_pct"), 5.0)
        cal  = float(np.clip(bias + mult * pre_cal, -abs(clip), abs(clip)))

        s = scores_map.get((t, reg))
        reliability = float(np.clip(s["score"] / 100.0, 0.25, 0.90)) if s else 0.50
        final = cal * reliability

        regime_val = _safe_float(r.get("regime"), 1.0)
        final, reginfo = _regime_shrink(final, macro, stock_regime=regime_val)
        final = float(np.clip(final, -5.0, 5.0))

        out_rows.append({
            "Ticker":          t,
            "Region":          reg,
            "Raw_ER_Pct":      round(raw, 4),
            "News_Delta_PP":   round(delta, 4),
            "PreCalib_ER_Pct": round(pre_cal, 4),
            "Calib_ER_Pct":    round(cal, 4),
            "Reliability":     round(reliability, 3),
            "RegimeShrink":    reginfo.get("shrink", 1.0),
            "volume_ratio":    round(_safe_float(r.get("volume_ratio"), 1.0), 3),
            "peer_rel_20d":    round(_safe_float(r.get("peer_rel_20d"), 0.0), 3),
            "regime":          int(regime_val),
            "score_0_100":     round(s["score"], 1) if s else 50.0,
            "confidence_tier": s["tier"] if s else "MEDIUM",
            "ER_Pct":          round(final, 4),
        })

    return pd.DataFrame(out_rows).sort_values(["Region","Ticker"]).reset_index(drop=True)
