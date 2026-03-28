from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from stockd import settings


def _scores_path():
    return getattr(settings, "SCORES_FILE", settings.DATA_DIR / "scores_stockd.csv")


@dataclass
class ScoreConfig:
    window_weeks: int = 52   # full year of history
    min_obs: int = 4          # start scoring after 4 weeks


def _clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def compute_scores(eval_df, cfg=None):
    """
    Compute per-ticker confidence scores from backtest eval data.
    Uses 52-week rolling window, penalises high MAE and systematic bias.
    Tiers: HIGH >=65, MEDIUM >=45, LOW <45
    """
    cfg = cfg or ScoreConfig()
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Ticker","Region","n","hit_rate","mae","bias","score_0_100","confidence_tier"])

    df = eval_df.copy()
    df["WeekStart"]    = pd.to_datetime(df["WeekStart"],    errors="coerce")
    df["AbsError_Pct"] = pd.to_numeric(df["AbsError_Pct"], errors="coerce")
    df["Error_Pct"]    = pd.to_numeric(df["Error_Pct"],    errors="coerce")
    df["DirectionHit"] = pd.to_numeric(df["DirectionHit"], errors="coerce")
    df = df.dropna(subset=["WeekStart","Ticker","Region","AbsError_Pct","Error_Pct"])

    out_rows = []
    for (tkr, reg), g in df.groupby(["Ticker","Region"], sort=False):
        g = g.sort_values("WeekStart").tail(cfg.window_weeks)
        n = int(len(g))
        if n == 0:
            continue

        hit_rate = float(np.nanmean(g["DirectionHit"].values)) if "DirectionHit" in g.columns else float("nan")
        mae  = float(np.nanmean(g["AbsError_Pct"].values))
        bias = float(np.nanmean(g["Error_Pct"].values))

        base = 50.0
        if not np.isnan(hit_rate):
            base += 30.0 * (hit_rate - 0.5)
        base -= min(mae * 0.8, 20.0)
        base -= min(abs(bias) * 0.5, 10.0)
        if n < cfg.min_obs:
            base -= 15.0 * (cfg.min_obs - n) / cfg.min_obs

        score = _clamp(base, 0.0, 100.0)

        if n < cfg.min_obs:   tier = "LOW"
        elif score >= 65:     tier = "HIGH"
        elif score >= 45:     tier = "MEDIUM"
        else:                 tier = "LOW"

        out_rows.append({
            "Ticker": str(tkr), "Region": str(reg),
            "n": n, "hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else None,
            "mae": round(mae, 4), "bias": round(bias, 4),
            "score_0_100": round(score, 2), "confidence_tier": tier,
        })

    return pd.DataFrame(out_rows)


def save_scores(scores_df):
    path = _scores_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(path, index=False)
    return path


def load_scores():
    path = _scores_path()
    if not path.exists():
        return pd.DataFrame(columns=["Ticker","Region","n","hit_rate","mae","bias","score_0_100","confidence_tier"])
    return pd.read_csv(path)
