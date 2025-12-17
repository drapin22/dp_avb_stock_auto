# stockd/evaluation.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stockd import settings


def load_prices(prices_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Expected columns: Date, Ticker, Region, Currency(optional), Close
    """
    if prices_path is None:
        default_path = getattr(settings, "PRICES_ALL_CSV", None)
        if default_path is not None and Path(default_path).exists():
            prices_path = default_path
        else:
            prices_path = settings.PRICES_HISTORY

    path = Path(prices_path)
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"{path} must contain Date and Ticker")
    if "Region" not in df.columns:
        df["Region"] = "UNKNOWN"
    if "Currency" not in df.columns:
        df["Currency"] = ""

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.strip()
    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    df = df[(df["Ticker"] != "") & (df["Close"] > 0)].copy()

    # IMPORTANT: sort by "on" first for merge_asof stability
    df = df.sort_values(["Date", "Ticker", "Region"]).reset_index(drop=True)

    # protect against scrape glitches (daily jump > 60%)
    df["pct_chg"] = df.groupby(["Ticker", "Region"], sort=False)["Close"].pct_change()
    df = df[df["pct_chg"].isna() | (df["pct_chg"].abs() <= 0.60)].copy()
    df = df.drop(columns=["pct_chg"], errors="ignore")

    return df


def dedup_forecasts(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep last forecast (by Date) for same (WeekStart, TargetDate, Ticker, Region, ModelVersion).
    """
    if forecasts_df is None or forecasts_df.empty:
        return pd.DataFrame(columns=[
            "Date", "WeekStart", "TargetDate", "ModelVersion",
            "Ticker", "Region", "HorizonDays", "ER_Pct", "Notes"
        ])

    df = forecasts_df.copy()

    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["Ticker", "Region", "ModelVersion"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["ER_Pct"] = pd.to_numeric(df.get("ER_Pct"), errors="coerce")

    df = df.dropna(subset=["Date", "WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"])
    df = df.sort_values("Date")

    key = ["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"]
    df = df.groupby(key, as_index=False).tail(1).reset_index(drop=True)
    return df


def load_forecasts(forecasts_path: Optional[str | Path] = None) -> pd.DataFrame:
    if forecasts_path is None:
        forecasts_path = settings.FORECASTS_FILE

    path = Path(forecasts_path)
    if not path.exists():
        return pd.DataFrame(columns=[
            "Date", "WeekStart", "TargetDate", "ModelVersion",
            "Ticker", "Region", "HorizonDays", "ER_Pct", "Notes"
        ])

    df = pd.read_csv(path)

    for col in ["Region", "Notes", "HorizonDays"]:
        if col not in df.columns:
            df[col] = np.nan

    return dedup_forecasts(df)


def evaluate_weekly(prices: Optional[pd.DataFrame] = None, forecasts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    start price = last close <= WeekStart
    end price   = last close <= TargetDate
    """
    prices = load_prices() if prices is None else prices.copy()
    forecasts = load_forecasts() if forecasts is None else forecasts.copy()

    if prices.empty or forecasts.empty:
        return pd.DataFrame()

    # normalize dtypes even if they were passed in
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip()
    prices["Region"] = prices["Region"].astype(str).str.strip()
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    forecasts["Date"] = pd.to_datetime(forecasts["Date"], errors="coerce")
    forecasts["WeekStart"] = pd.to_datetime(forecasts["WeekStart"], errors="coerce")
    forecasts["TargetDate"] = pd.to_datetime(forecasts["TargetDate"], errors="coerce")
    forecasts["Ticker"] = forecasts["Ticker"].astype(str).str.strip()
    forecasts["Region"] = forecasts["Region"].astype(str).str.strip()
    forecasts["ModelVersion"] = forecasts["ModelVersion"].astype(str).str.strip()
    forecasts["ER_Pct"] = pd.to_numeric(forecasts.get("ER_Pct"), errors="coerce")

    prices = prices.dropna(subset=["Date", "Ticker", "Region", "Close"])
    forecasts = forecasts.dropna(subset=["Date", "WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion", "ER_Pct"])

    today_ts = pd.Timestamp(date.today())
    eligible = forecasts[forecasts["TargetDate"] < today_ts].copy()
    if eligible.empty:
        return pd.DataFrame()

    p = prices[["Date", "Ticker", "Region", "Close"]].copy()
    p = p.sort_values(["Date", "Ticker", "Region"]).reset_index(drop=True)

    # CRITICAL: sort left by on-key FIRST so it's globally monotonic for merge_asof
    left = eligible.sort_values(["WeekStart", "Ticker", "Region"]).reset_index(drop=True)

    start = pd.merge_asof(
        left,
        p,
        left_on="WeekStart",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="backward",
        suffixes=("", "_px_start"),
    ).rename(columns={"Date_px_start": "PriceDateStart", "Close": "CloseStart"})

    # sort by TargetDate first for the second asof merge
    start = start.sort_values(["TargetDate", "Ticker", "Region"]).reset_index(drop=True)

    end = pd.merge_asof(
        start,
        p,
        left_on="TargetDate",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="backward",
        suffixes=("", "_px_end"),
    ).rename(columns={"Date_px_end": "PriceDateEnd", "Close": "CloseEnd"})

    df = end.copy()
    df["Model_ER_Pct"] = pd.to_numeric(df.get("ER_Pct"), errors="coerce")
    df["CloseStart"] = pd.to_numeric(df.get("CloseStart"), errors="coerce")
    df["CloseEnd"] = pd.to_numeric(df.get("CloseEnd"), errors="coerce")

    df = df.dropna(subset=["Model_ER_Pct", "CloseStart", "CloseEnd"])
    df = df[(df["CloseStart"] > 0) & (df["CloseEnd"] > 0)].copy()

    df["Realized_Pct"] = (df["CloseEnd"] / df["CloseStart"] - 1.0) * 100.0
    df["Error_Pct"] = df["Realized_Pct"] - df["Model_ER_Pct"]
    df["AbsError_Pct"] = df["Error_Pct"].abs()

    df["DirectionHit"] = 0
    mask = (df["Realized_Pct"].abs() > 1e-12) & (df["Model_ER_Pct"].abs() > 1e-12)
    df.loc[mask, "DirectionHit"] = (
        np.sign(df.loc[mask, "Realized_Pct"]) == np.sign(df.loc[mask, "Model_ER_Pct"])
    ).astype(int)

    out_cols = [
        "WeekStart", "TargetDate", "Date", "ModelVersion", "Ticker", "Region", "HorizonDays",
        "Model_ER_Pct", "CloseStart", "CloseEnd", "Realized_Pct", "Error_Pct", "AbsError_Pct",
        "DirectionHit", "Notes"
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[out_cols].copy()


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=[
            "WeekStart", "ModelVersion", "Region", "n", "hit_rate", "mean_abs_error", "mean_error", "median_abs_error"
        ])

    g = eval_df.copy()
    g["AbsError_Pct"] = pd.to_numeric(g["AbsError_Pct"], errors="coerce")
    g["Error_Pct"] = pd.to_numeric(g["Error_Pct"], errors="coerce")
    g["DirectionHit"] = pd.to_numeric(g["DirectionHit"], errors="coerce")

    grp = g.groupby(["WeekStart", "ModelVersion", "Region"], dropna=False)
    res = grp.agg(
        n=("Ticker", "count"),
        hit_rate=("DirectionHit", "mean"),
        mean_abs_error=("AbsError_Pct", "mean"),
        mean_error=("Error_Pct", "mean"),
        median_abs_error=("AbsError_Pct", "median"),
    ).reset_index()

    return res
