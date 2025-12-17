# stockd/calibration.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from stockd import settings


def _calib_path() -> Path:
    p = getattr(settings, "CALIBRATION_FILE", None)
    if p is None:
        return settings.DATA_DIR / "calibration.json"
    return Path(p)


def load_calibration() -> Dict[str, Any]:
    path = _calib_path()
    if not path.exists():
        return {
            "version": 2,
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
    """
    Fit y ≈ a + b*x (ridge pe b, a nepenalizat).
    lam = strength (mai mare => b mai aproape de 1 prin shrink implicit)
    """
    if len(x) < 5:
        return 0.0, 1.0  # bias=0, mult=1

    x = x.astype(float)
    y = y.astype(float)

    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean

    denom = float((xc * xc).sum() + lam)
    if denom <= 1e-12:
        b = 1.0
    else:
        b = float((xc * yc).sum() / denom)

    a = float(y_mean - b * x_mean)
    return a, b


def build_region_calibration(
    eval_df: pd.DataFrame,
    prior_strength: float = 50.0,
) -> Dict[str, Any]:
    """
    Determinist: învață bias/mult pe regiune din istoricul săptămânilor evaluate.
    prior_strength = ridge lambda (mai mare => update mai conservator)
    """
    base = load_calibration()

    if eval_df is None or eval_df.empty:
        base["notes"] = "no eval rows, kept previous calibration"
        return base

    df = eval_df.copy()
    df["Region"] = df["Region"].astype(str)
    df["Model_ER_Pct"] = pd.to_numeric(df["Model_ER_Pct"], errors="coerce")
    df["Realized_Pct"] = pd.to_numeric(df["Realized_Pct"], errors="coerce")
    df = df.dropna(subset=["Region", "Model_ER_Pct", "Realized_Pct"])

    for region, g in df.groupby("Region"):
        x = g["Model_ER_Pct"].values
        y = g["Realized_Pct"].values
        a, b = _ridge_fit_slope_intercept(x, y, lam=prior_strength)

        # garduri: nu lăsăm mult/bias să sară prea tare
        b = _clamp(b, 0.25, 2.50)
        a = _clamp(a, -5.0, 5.0)

        if "regions" not in base:
            base["regions"] = {}
        if region not in base["regions"]:
            base["regions"][region] = {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0}

        base["regions"][region]["mult"] = float(b)
        base["regions"][region]["bias"] = float(a)
        base["regions"][region]["n"] = int(len(g))

    base["notes"] = "updated from eval_df via ridge fit"
    return base


def merge_coach_suggestions(
    calib: Dict[str, Any],
    coach: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aplică sugestiile LLM doar în limite stricte.
    Nu atinge mult/bias direct (decât prin build_region_calibration determinist).
    """
    out = json.loads(json.dumps(calib))  # deep copy

    if not coach or coach.get("_coach_status") != "OK":
        out["notes"] = (out.get("notes", "") + " | coach: skipped").strip()
        return out

    alpha = coach.get("alpha", out.get("alpha", 0.20))
    out["alpha"] = _clamp(float(alpha), 0.05, 0.40)

    regions = coach.get("regions", {}) or {}
    out.setdefault("regions", {})
    for reg, cfg in regions.items():
        if reg not in out["regions"]:
            out["regions"][reg] = {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0, "n": 0}
        clip_pct = cfg.get("clip_pct", out["regions"][reg].get("clip_pct", 6.0))
        out["regions"][reg]["clip_pct"] = _clamp(float(clip_pct), 1.0, 10.0)

    # ticker caps: doar cap, nu “predicții”
    caps = coach.get("ticker_overrides", []) or []
    out.setdefault("ticker_caps", {})
    for item in caps:
        t = str(item.get("Ticker", "")).strip()
        r = str(item.get("Region", "")).strip()
        cap = item.get("multiplier_cap", None)
        if not t or not r or cap is None:
            continue
        cap = _clamp(float(cap), 0.25, 2.50)
        out["ticker_caps"][f"{t}::{r}"] = cap

    out["notes"] = (out.get("notes", "") + " | coach: applied safe suggestions").strip()
    return out


def apply_calibration(pred_df: pd.DataFrame, calib: Dict[str, Any]) -> pd.DataFrame:
    """
    pred_df trebuie să aibă: Ticker, Region, ER_Pct
    Returnează: Adj_ER_Pct + info coloane
    """
    df = pred_df.copy()
    df["Ticker"] = df["Ticker"].astype(str)
    df["Region"] = df["Region"].astype(str)
    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    regions = (calib.get("regions") or {})
    ticker_caps = (calib.get("ticker_caps") or {})

    adj = []
    mult_used = []
    bias_used = []
    clip_used = []
    cap_used = []

    for _, r in df.iterrows():
        reg = r["Region"]
        tkr = r["Ticker"]
        er = float(r["ER_Pct"])

        rcfg = regions.get(reg, {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0})
        mult = float(rcfg.get("mult", 1.0))
        bias = float(rcfg.get("bias", 0.0))
        clip_pct = float(rcfg.get("clip_pct", 6.0))

        # ticker cap pe mult (opțional)
        cap_key = f"{tkr}::{reg}"
        cap = ticker_caps.get(cap_key, None)
        if cap is not None:
            mult = _clamp(mult, 0.25, float(cap))
            cap_used.append(float(cap))
        else:
            cap_used.append(float("nan"))

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
    return df
