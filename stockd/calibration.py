# stockd/calibration.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from stockd import settings


def _fit_linear(pred: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    # OLS: real = alpha + beta * pred
    if len(pred) < 3:
        return 0.0, 1.0
    v = float(np.var(pred))
    if v <= 1e-12:
        return 0.0, 1.0
    beta = float(np.cov(pred, real, ddof=0)[0, 1] / v)
    alpha = float(np.mean(real) - beta * np.mean(pred))
    return alpha, beta


def build_region_calibration(eval_df: pd.DataFrame) -> dict:
    """
    Învață calibrare pe regiune (alpha/beta) din evaluare.
    Include "usable" ca să nu aplicăm calibrare degenerată.
    """
    if eval_df is None or eval_df.empty:
        return {"version": 2, "regions": {}, "notes": "no data", "model_version_tag": settings.MODEL_VERSION_TAG}

    regions = {}
    for region, g in eval_df.groupby("Region"):
        pred = g["Model_ER_Pct"].astype(float).to_numpy()
        real = g["Realized_Pct"].astype(float).to_numpy()

        n = int(len(pred))
        mae = float(np.mean(np.abs(real - pred))) if n else float("nan")

        std_pred = float(np.std(pred)) if n else 0.0
        std_real = float(np.std(real)) if n else 0.0

        # winsorize pentru robustete, doar dacă avem destule puncte
        if n >= 20:
            p_lo, p_hi = np.quantile(pred, [0.05, 0.95])
            r_lo, r_hi = np.quantile(real, [0.05, 0.95])
            pred_c = np.clip(pred, p_lo, p_hi)
            real_c = np.clip(real, r_lo, r_hi)
        else:
            pred_c = pred
            real_c = real

        alpha, beta = _fit_linear(pred_c, real_c)

        # Guardrails: dacă pred e aproape constant sau n prea mic, calibrarea devine “offset-only”
        usable = True
        reasons = []

        if n < 12:
            usable = False
            reasons.append("too_few_points")

        if std_pred < 0.10:
            usable = False
            reasons.append("pred_variance_too_low")

        # dacă beta e aproape 0, practic doar alpha mută totul
        if abs(beta) < 0.20 and abs(alpha) > 0.30:
            usable = False
            reasons.append("degenerate_beta_offset_only")

        # Safety clamp pentru alpha/beta
        alpha = float(np.clip(alpha, -3.0, 3.0))
        beta = float(np.clip(beta, 0.0, 2.0))

        regions[region] = {
            "alpha": alpha,
            "beta": beta,
            "mae": float(mae),
            "n": n,
            "std_pred": float(std_pred),
            "std_real": float(std_real),
            "usable": bool(usable),
            "reasons": reasons,
        }

    return {
        "version": 2,
        "regions": regions,
        "model_version_tag": settings.MODEL_VERSION_TAG,
        "notes": "region-level calibration with usability gates",
    }


def save_calibration(calib: dict) -> None:
    settings.CALIBRATION_FILE.write_text(json.dumps(calib, indent=2), encoding="utf-8")


def load_calibration() -> dict:
    if not settings.CALIBRATION_FILE.exists():
        return {"version": 2, "regions": {}}
    try:
        return json.loads(settings.CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 2, "regions": {}}


def apply_calibration(forecasts_df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    Adaugă:
      - Adj_ER_Pct
      - CalibApplied (True/False)
      - CalibReason (string)
    """
    df = forecasts_df.copy()
    if df.empty:
        df["Adj_ER_Pct"] = df.get("ER_Pct", 0.0)
        df["CalibApplied"] = False
        df["CalibReason"] = "empty"
        return df

    regions = calib.get("regions", {}) if isinstance(calib, dict) else {}

    adj_vals = []
    applied = []
    reason = []

    for _, r in df.iterrows():
        region = str(r.get("Region", "")).strip()
        raw = float(r.get("ER_Pct", 0.0))

        p = regions.get(region)
        if not p:
            adj_vals.append(raw)
            applied.append(False)
            reason.append("no_region_params")
            continue

        usable = bool(p.get("usable", True))
        if not usable:
            adj_vals.append(raw)
            applied.append(False)
            reason.append("calib_not_usable:" + ",".join(p.get("reasons", [])))
            continue

        alpha = float(p.get("alpha", 0.0))
        beta = float(p.get("beta", 1.0))
        adj = alpha + beta * raw

        # clamp per region
        max_abs = 8.0 if region in {"RO", "EU"} else 12.0
        adj = float(np.clip(adj, -max_abs, max_abs))

        adj_vals.append(adj)
        applied.append(True)
        reason.append("ok")

    df["Adj_ER_Pct"] = adj_vals
    df["CalibApplied"] = applied
    df["CalibReason"] = reason
    return df
