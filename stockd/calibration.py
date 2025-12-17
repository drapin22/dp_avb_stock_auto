# stockd/calibration.py
from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import pandas as pd

from stockd import settings


@dataclass
class CalibParams:
    alpha: float
    beta: float
    mae: float
    n: int


def _fit_linear(pred: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    # OLS: real = alpha + beta * pred
    if len(pred) < 3:
        return 0.0, 1.0
    v = np.var(pred)
    if v <= 1e-9:
        return float(np.mean(real)), 0.0
    beta = float(np.cov(pred, real, ddof=0)[0, 1] / v)
    alpha = float(np.mean(real) - beta * np.mean(pred))
    return alpha, beta


def build_region_calibration(eval_df: pd.DataFrame) -> dict:
    """
    Învață calibrare pe regiune pe baza eval_df (weekly).
    Scrie parametrii în dict serializabil JSON.
    """
    if eval_df.empty:
        return {"version": 1, "regions": {}, "notes": "no data"}

    regions = {}
    for region, g in eval_df.groupby("Region"):
        pred = g["Model_ER_Pct"].astype(float).values
        real = g["Realized_Pct"].astype(float).values

        # clipping robust: winsorize la 5/95
        p_lo, p_hi = np.quantile(pred, [0.05, 0.95]) if len(pred) >= 20 else (np.min(pred), np.max(pred))
        r_lo, r_hi = np.quantile(real, [0.05, 0.95]) if len(real) >= 20 else (np.min(real), np.max(real))
        pred_c = np.clip(pred, p_lo, p_hi)
        real_c = np.clip(real, r_lo, r_hi)

        alpha, beta = _fit_linear(pred_c, real_c)

        mae = float(np.mean(np.abs(real - pred))) if len(pred) else float("nan")
        n = int(len(pred))

        regions[region] = {
            "alpha": float(alpha),
            "beta": float(beta),
            "mae": float(mae),
            "n": n,
        }

    return {
        "version": 1,
        "regions": regions,
        "model_version_tag": settings.MODEL_VERSION_TAG,
        "notes": "region-level linear calibration (winsorized)",
    }


def save_calibration(calib: dict) -> None:
    settings.CALIBRATION_FILE.write_text(json.dumps(calib, indent=2), encoding="utf-8")


def load_calibration() -> dict:
    if not settings.CALIBRATION_FILE.exists():
        return {"version": 1, "regions": {}}
    try:
        return json.loads(settings.CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "regions": {}}


def apply_calibration(forecasts_df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    Adaugă:
      - Adj_ER_Pct
      - CalibAlpha, CalibBeta, CalibMAE, CalibN
    """
    df = forecasts_df.copy()
    if df.empty:
        df["Adj_ER_Pct"] = df.get("ER_Pct", 0.0)
        return df

    regions = calib.get("regions", {}) if isinstance(calib, dict) else {}

    adj_vals = []
    alphas = []
    betas = []
    maes = []
    ns = []

    for _, r in df.iterrows():
        region = str(r.get("Region", "")).strip()
        raw = float(r.get("ER_Pct", 0.0))
        p = regions.get(region)

        if not p:
            adj = raw
            alpha, beta, mae, n = 0.0, 1.0, float("nan"), 0
        else:
            alpha = float(p.get("alpha", 0.0))
            beta = float(p.get("beta", 1.0))
            mae = float(p.get("mae", float("nan")))
            n = int(p.get("n", 0))
            adj = alpha + beta * raw

            # shrink dacă avem puține date / mae mare
            if n < 10 and np.isfinite(mae):
                w = max(0.3, min(1.0, n / 20))
                adj = w * adj + (1 - w) * raw
            if np.isfinite(mae) and mae > 5:
                adj *= 0.8

        # safety clamp per region (evită “explozia” din calibrare)
        max_abs = 8.0 if region in {"RO", "EU"} else 12.0
        adj = float(np.clip(adj, -max_abs, max_abs))

        adj_vals.append(adj)
        alphas.append(alpha)
        betas.append(beta)
        maes.append(mae)
        ns.append(n)

    df["Adj_ER_Pct"] = adj_vals
    df["CalibAlpha"] = alphas
    df["CalibBeta"] = betas
    df["CalibMAE"] = maes
    df["CalibN"] = ns
    return df
