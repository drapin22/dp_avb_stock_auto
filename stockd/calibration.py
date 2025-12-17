# stockd/calibration.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from stockd import settings


def _calib_path() -> Path:
    return getattr(settings, "CALIBRATION_FILE", settings.DATA_DIR / "calibration.json")


def _mentor_overrides_path() -> Path:
    return getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")


def load_calibration() -> Dict[str, Any]:
    path = _calib_path()
    if not path.exists():
        return {
            "version": 3,
            "updated_at": None,
            "alpha": 0.20,
            "regions": {
                "RO": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0},
                "EU": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0},
                "US": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0},
            },
            "ticker_caps": {},
            "notes": "bootstrap calibration",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_calibration(calib: Dict[str, Any]) -> None:
    path = _calib_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    calib["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _ridge_fit_slope_intercept(x: np.ndarray, y: np.ndarray, lam: float) -> Tuple[float, float]:
    if len(x) < 6:
        return 0.0, 1.0
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    denom = float((xc * xc).sum() + lam)
    b = float((xc * yc).sum() / denom) if denom > 1e-12 else 1.0
    a = float(y_mean - b * x_mean)
    return a, b


def build_region_calibration(eval_df: pd.DataFrame, prior_strength: float = 50.0) -> Dict[str, Any]:
    base = load_calibration()
    if eval_df is None or eval_df.empty:
        base["notes"] = "no eval rows, kept previous calibration"
        return base

    df = eval_df.copy()
    df["Region"] = df["Region"].astype(str)
    df["Model_ER_Pct"] = pd.to_numeric(df["Model_ER_Pct"], errors="coerce")
    df["Realized_Pct"] = pd.to_numeric(df["Realized_Pct"], errors="coerce")
    df = df.dropna(subset=["Region", "Model_ER_Pct", "Realized_Pct"])

    base.setdefault("regions", {})
    for region, g in df.groupby("Region"):
        x = g["Model_ER_Pct"].values
        y = g["Realized_Pct"].values
        a, b = _ridge_fit_slope_intercept(x, y, lam=prior_strength)

        b = _clamp(b, 0.25, 2.50)
        a = _clamp(a, -5.0, 5.0)

        base["regions"].setdefault(region, {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0})
        base["regions"][region]["mult"] = float(b)
        base["regions"][region]["bias"] = float(a)
        base["regions"][region]["n"] = int(len(g))

    base["notes"] = "updated from eval_df via ridge fit"
    return base


def _load_mentor_overrides() -> Dict[str, Any]:
    path = _mentor_overrides_path()
    if not path.exists():
        return {"status": "MISSING", "items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "INVALID", "items": []}


def apply_calibration(pred_df: pd.DataFrame, calib: Dict[str, Any]) -> pd.DataFrame:
    df = pred_df.copy()
    df["Ticker"] = df["Ticker"].astype(str)
    df["Region"] = df["Region"].astype(str)
    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    regions = (calib.get("regions") or {})
    ticker_caps = (calib.get("ticker_caps") or {})

    mentor = _load_mentor_overrides()
    mentor_items = mentor.get("items", []) or []
    mentor_map: Dict[str, Dict[str, Any]] = {}
    for it in mentor_items:
        t = str(it.get("Ticker", "")).strip()
        r = str(it.get("Region", "")).strip()
        if t and r:
            mentor_map[f"{t}::{r}"] = it

    adj = []
    mult_used = []
    bias_used = []
    clip_used = []
    cap_used = []
    mentor_clip_used = []
    mentor_cap_used = []

    for _, r in df.iterrows():
        reg = r["Region"]
        tkr = r["Ticker"]
        er = float(r["ER_Pct"])

        rcfg = regions.get(reg, {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0})
        mult = float(rcfg.get("mult", 1.0))
        bias = float(rcfg.get("bias", 0.0))
        clip_pct = float(rcfg.get("clip_pct", 6.0))

        # cap din calibrare (manual/learning)
        cap_key = f"{tkr}::{reg}"
        cap = ticker_caps.get(cap_key, None)
        if cap is not None:
            mult = _clamp(mult, 0.25, float(cap))
            cap_used.append(float(cap))
        else:
            cap_used.append(float("nan"))

        # overrides de la mentor (safe)
        m = mentor_map.get(cap_key)
        m_clip = None
        m_cap = None
        if m:
            if "clip_pct" in m:
                m_clip = _clamp(float(m["clip_pct"]), 1.0, 10.0)
                clip_pct = min(clip_pct, m_clip)  # doar mai conservator
            if "multiplier_cap" in m:
                m_cap = _clamp(float(m["multiplier_cap"]), 0.25, 2.50)
                mult = min(mult, m_cap)  # doar mai conservator

        mentor_clip_used.append(m_clip if m_clip is not None else float("nan"))
        mentor_cap_used.append(m_cap if m_cap is not None else float("nan"))

        adj_er = bias + mult * er
        adj_er = _clamp(adj_er, -clip_pct, clip_pct)

        adj.append(float(adj_er))
        mult_used.append(float(mult))
        bias_used.append(float(bias))
        clip_used.append(float(clip_pct))

    df["Adj_ER_Pct"] = adj
    df["CalibApplied"] = True
    df["Calib_mult"] = mult_used
    df["Calib_bias"] = bias_used
    df["Calib_clip_pct"] = clip_used
    df["Calib_mult_cap"] = cap_used
    df["Mentor_clip_pct_used"] = mentor_clip_used
    df["Mentor_mult_cap_used"] = mentor_cap_used
    df["Mentor_overrides_status"] = str(mentor.get("status", "MISSING"))
    return df
