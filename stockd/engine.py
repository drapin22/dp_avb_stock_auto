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


def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _regime_shrink(er: float, macro: Dict) -> Tuple[float, Dict]:
    """Shrink forecasts under 'risk-off' style regimes."""
    vix = _safe_float(macro.get("vix_level"), 0.0)
    dxy = _safe_float(macro.get("dxy_ret_5d"), 0.0)

    shrink = 1.0
    if vix >= 25:
        shrink *= 0.75
    if dxy >= 1.0:
        shrink *= 0.85

    return er * shrink, {"shrink": shrink, "vix": vix, "dxy": dxy}


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date | None = None,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """
    Returns a DataFrame with at least:
      - Ticker, Region, ER_Pct
    Also returns debug columns:
      - Raw_ER_Pct, PreCalib_ER_Pct, Calib_ER_Pct, News_Delta_PP, Reliability, RegimeShrink
    """
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
    # Add macro columns expected by the model
    for k in ["vix_level", "dxy_ret_5d", "oil_ret_5d", "gold_ret_5d", "bench_ret_5d"]:
        model_in[k] = _safe_float(macro.get(k), 0.0)

    # For RO: if bench_ret_5d isn't available, use RO proxy
    if "ro_proxy_ret_5d" in model_in.columns:
        mask_ro = model_in["Region"] == "RO"
        missing_bench = mask_ro & (pd.to_numeric(model_in["bench_ret_5d"], errors="coerce").fillna(0.0) == 0.0)
        model_in.loc[missing_bench, "bench_ret_5d"] = pd.to_numeric(model_in.loc[missing_bench, "ro_proxy_ret_5d"], errors="coerce").fillna(0.0)

    state = load_state()
    base_pred = predict(model_in, state=state)

    # Optional LLM news deltas
    deltas = {}
    if settings.ENABLE_LLM_NEWS_ADJ and settings.OPENAI_API_KEY:
        try:
            deltas = propose_news_deltas(
                tickers=model_in[["Ticker", "Region"]].to_dict(orient="records"),
                macro=macro,
                as_of=as_of,
            )
        except Exception:
            deltas = {}

    # Load calibration + scores
    calib = load_calibration()
    calib_regions = (calib or {}).get("regions", {})
    scores = load_scores()
    scores_map = {}
    if scores is not None and not scores.empty:
        for _, r in scores.iterrows():
            scores_map[(str(r["Ticker"]).upper().strip(), str(r["Region"]).upper().strip())] = {
                "score": _safe_float(r.get("score_0_100"), 50.0),
                "tier": str(r.get("confidence_tier") or "").strip(),
            }

    out_rows = []
    for i, r in model_in.iterrows():
        t = r["Ticker"]
        reg = r["Region"]
        raw = _safe_float(base_pred[i], 0.0)
        delta = _safe_float(deltas.get((t, reg), 0.0), 0.0)

        pre_cal = raw + delta

        # calibration per region
        rcfg = calib_regions.get(reg, {})
        mult = _safe_float(rcfg.get("mult"), 1.0)
        bias = _safe_float(rcfg.get("bias"), 0.0)
        clip = _safe_float(rcfg.get("clip_pct"), 5.0)
        cal = bias + mult * pre_cal
        cal = float(np.clip(cal, -abs(clip), abs(clip)))

        # reliability shrink from scores
        s = scores_map.get((t, reg))
        if s:
            reliability = float(np.clip(s["score"] / 100.0, 0.25, 0.90))
        else:
            reliability = 0.50

        final = cal * reliability

        # regime shrink (risk-off)
        final, reginfo = _regime_shrink(final, macro)
        final = float(np.clip(final, -5.0, 5.0))

        out_rows.append(
            {
                "Ticker": t,
                "Region": reg,
                "Raw_ER_Pct": raw,
                "News_Delta_PP": delta,
                "PreCalib_ER_Pct": pre_cal,
                "Calib_ER_Pct": cal,
                "Reliability": reliability,
                "RegimeShrink": reginfo.get("shrink", 1.0),
                "ER_Pct": final,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["Region", "Ticker"]).reset_index(drop=True)
