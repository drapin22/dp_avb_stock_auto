from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Tuple

import pandas as pd

from stockd import settings


REQUIRED_PRICES_COLS = ["Date", "Ticker", "Region", "Close"]
REQUIRED_FORECASTS_COLS = ["WeekStart", "TargetDate", "ModelVersion", "Ticker", "Region", "HorizonDays", "ER_Pct"]


def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_prices(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.PRICES_HISTORY
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)
    df = _ensure_dt(df, "Date")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.upper().str.strip()
    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    return df.sort_values(["Ticker", "Region", "Date"]).reset_index(drop=True)


def load_forecasts(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.FORECASTS_FILE
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_FORECASTS_COLS)

    df = pd.read_csv(path)
    df = _ensure_dt(df, "Date")
    df = _ensure_dt(df, "WeekStart")
    df = _ensure_dt(df, "TargetDate")

    for c in ["Ticker", "Region", "ModelVersion"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()

    if "ER_Pct" in df.columns:
        df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce")

    if "HorizonDays" in df.columns:
        df["HorizonDays"] = pd.to_numeric(df["HorizonDays"], errors="coerce")

    df = df.dropna(subset=["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"])
    return df.reset_index(drop=True)


def dedup_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep last forecast per Ticker/Region/TargetDate/ModelVersion."""
    f = forecasts.copy()
    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in f.columns:
            f[c] = pd.to_datetime(f[c], errors="coerce")

    f = f.sort_values(["Ticker", "Region", "TargetDate", "ModelVersion", "Date"], na_position="last")
    f = f.drop_duplicates(subset=["Ticker", "Region", "TargetDate", "ModelVersion"], keep="last")
    return f.reset_index(drop=True)


def _eval_window(prices: pd.DataFrame, forecasts_win: pd.DataFrame, ws: pd.Timestamp, we: pd.Timestamp) -> pd.DataFrame:
    """Compute realized return for each ticker between ws..we (inclusive)."""
    p = prices.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p = p.dropna(subset=["Date", "Ticker", "Region", "Close"])
    p = p[(p["Date"] >= ws) & (p["Date"] <= we)].sort_values(["Ticker", "Region", "Date"])

    if p.empty:
        out = forecasts_win.copy()
        out["StartClose"] = pd.NA
        out["EndClose"] = pd.NA
        out["Realized_Pct"] = pd.NA
        return out

    g = p.groupby(["Ticker", "Region"], as_index=False)
    first = g.first()[["Ticker", "Region", "Date", "Close"]].rename(columns={"Date": "StartDate", "Close": "StartClose"})
    last = g.last()[["Ticker", "Region", "Date", "Close"]].rename(columns={"Date": "EndDate", "Close": "EndClose"})

    rr = first.merge(last, on=["Ticker", "Region"], how="inner")
    rr["Realized_Pct"] = (rr["EndClose"] / rr["StartClose"] - 1.0) * 100.0

    out = forecasts_win.merge(rr[["Ticker", "Region", "StartClose", "EndClose", "Realized_Pct"]], on=["Ticker", "Region"], how="left")
    return out


def evaluate_weekly(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Evaluate forecasts for each unique (WeekStart, TargetDate) window."""
    if prices.empty or forecasts.empty:
        return pd.DataFrame()

    f = forecasts.copy()
    f["WeekStart"] = pd.to_datetime(f["WeekStart"], errors="coerce")
    f["TargetDate"] = pd.to_datetime(f["TargetDate"], errors="coerce")
    f = f.dropna(subset=["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"])

    if f.empty:
        return pd.DataFrame()

    # Ensure numeric
    f["ER_Pct"] = pd.to_numeric(f.get("ER_Pct"), errors="coerce")

    out_frames = []
    for (ws, we), grp in f.groupby(["WeekStart", "TargetDate"], dropna=True):
        out_frames.append(_eval_window(prices, grp, ws=ws, we=we))

    out = pd.concat(out_frames, ignore_index=True)

    out["Model_ER_Pct"] = pd.to_numeric(out.get("ER_Pct"), errors="coerce")
    out["Realized_Pct"] = pd.to_numeric(out.get("Realized_Pct"), errors="coerce")

    out["Error_Pct"] = out["Model_ER_Pct"] - out["Realized_Pct"]
    out["AbsError_Pct"] = out["Error_Pct"].abs()
    out["DirectionHit"] = (out["Model_ER_Pct"].fillna(0) * out["Realized_Pct"].fillna(0) > 0).astype(int)

    cols = [
        "WeekStart", "TargetDate", "ModelVersion",
        "Ticker", "Region", "HorizonDays",
        "Model_ER_Pct", "Realized_Pct",
        "Error_Pct", "AbsError_Pct", "DirectionHit",
        "StartClose", "EndClose",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA

    return out[cols].sort_values(["TargetDate", "Region", "Ticker"]).reset_index(drop=True)


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Region", "Count", "MAE_Pct", "HitRate"])

    g = eval_df.groupby("Region", dropna=False)
    summary = g.agg(
        Count=("Ticker", "count"),
        MAE_Pct=("AbsError_Pct", "mean"),
        HitRate=("DirectionHit", "mean"),
    ).reset_index()

    # Total row
    total = pd.DataFrame([{
        "Region": "ALL",
        "Count": int(eval_df.shape[0]),
        "MAE_Pct": float(eval_df["AbsError_Pct"].mean()),
        "HitRate": float(eval_df["DirectionHit"].mean()),
    }])
    return pd.concat([summary, total], ignore_index=True)
