# stockd/calibration.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from stockd import settings


def load_calibration() -> dict:
    if not settings.CALIBRATION_FILE.exists():
        return {
            "version": 1,
            "regions": {
                "RO": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0},
                "EU": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0},
                "US": {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0},
            },
            "notes": "bootstrap"
        }
    return json.loads(settings.CALIBRATION_FILE.read_text(encoding="utf-8"))


def save_calibration(calib: dict) -> None:
    settings.CALIBRATION_FILE.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")


def build_region_calibration(eval_df: pd.DataFrame, ridge: float = 50.0) -> dict:
    calib = load_calibration()
    calib.setdefault("regions", {})

    if eval_df is None or eval_df.empty:
        calib["notes"] = "no eval rows"
        return calib

    df = eval_df.copy()
    df["Model_ER_Pct"] = pd.to_numeric(df["Model_ER_Pct"], errors="coerce")
    df["Realized_Pct"] = pd.to_numeric(df["Realized_Pct"], errors="coerce")
    df = df.dropna(subset=["Region","Model_ER_Pct","Realized_Pct"])

    for region, g in df.groupby("Region"):
        x = g["Model_ER_Pct"].values.astype(float)
        y = g["Realized_Pct"].values.astype(float)
        if len(x) < 6:
            a, b = 0.0, 1.0
        else:
            xm, ym = x.mean(), y.mean()
            xc, yc = x - xm, y - ym
            denom = float((xc*xc).sum() + ridge)
            b = float((xc*yc).sum() / denom) if denom > 1e-12 else 1.0
            a = float(ym - b*xm)

        b = float(max(0.25, min(2.50, b)))
        a = float(max(-5.0, min(5.0, a)))

        calib["regions"].setdefault(region, {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0})
        calib["regions"][region]["mult"] = b
        calib["regions"][region]["bias"] = a

    calib["notes"] = "ridge fit per region"
    return calib


def apply_calibration(pred_df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    df = pred_df.copy()
    df["Ticker"] = df["Ticker"].astype(str)
    df["Region"] = df["Region"].astype(str)
    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    regions = calib.get("regions", {})
    mentor = {"status":"MISSING","items":[]}
    if settings.MENTOR_OVERRIDES_FILE.exists():
        try:
            mentor = json.loads(settings.MENTOR_OVERRIDES_FILE.read_text(encoding="utf-8"))
        except Exception:
            mentor = {"status":"INVALID","items":[]}

    mentor_map = {}
    for it in mentor.get("items", []) or []:
        t = str(it.get("Ticker","")).strip()
        r = str(it.get("Region","")).strip()
        if t and r:
            mentor_map[f"{t}::{r}"] = it

    adj = []
    for _, row in df.iterrows():
        reg = row["Region"]
        tkr = row["Ticker"]
        er = float(row["ER_Pct"])

        rcfg = regions.get(reg, {"mult": 1.0, "bias": 0.0, "clip_pct": 6.0})
        mult = float(rcfg.get("mult", 1.0))
        bias = float(rcfg.get("bias", 0.0))
        clip = float(rcfg.get("clip_pct", 6.0))

        m = mentor_map.get(f"{tkr}::{reg}")
        if m:
            if "clip_pct" in m:
                clip = min(clip, float(m["clip_pct"]))
            if "multiplier_cap" in m:
                mult = min(mult, float(m["multiplier_cap"]))

        a = bias + mult * er
        a = float(max(-clip, min(clip, a)))
        adj.append(a)

    df["Adj_ER_Pct"] = adj
    df["Mentor_overrides_status"] = str(mentor.get("status","MISSING"))
    return df
