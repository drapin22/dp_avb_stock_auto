# stockd/evaluation.py
from __future__ import annotations

from datetime import date
import pandas as pd
import numpy as np

from stockd import settings


def load_prices() -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    return df


def load_forecasts() -> pd.DataFrame:
    if not settings.FORECASTS_FILE.exists():
        return pd.DataFrame(columns=[
            "Date","WeekStart","TargetDate","ModelVersion","Ticker","Region","HorizonDays","ER_Pct","Notes"
        ])
    df = pd.read_csv(settings.FORECASTS_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["WeekStart"] = pd.to_datetime(df["WeekStart"], errors="coerce")
    df["TargetDate"] = pd.to_datetime(df["TargetDate"], errors="coerce")
    df["ER_Pct"] = pd.to_numeric(df.get("ER_Pct", 0.0), errors="coerce").fillna(0.0)
    if "ModelVersion" not in df.columns:
        df["ModelVersion"] = settings.MODEL_VERSION_TAG
    df = df.dropna(subset=["Date", "WeekStart", "TargetDate", "Ticker", "Region"])
    return df


def dedup_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts
    forecasts = forecasts.sort_values("Date", kind="mergesort")
    key = ["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"]
    forecasts = forecasts.groupby(key, as_index=False).tail(1).reset_index(drop=True)
    return forecasts


def _sort_for_asof(df: pd.DataFrame, on_col: str, by_cols: list[str]) -> pd.DataFrame:
    """
    merge_asof în pandas cere, în practică, sortare globală după on_col.
    Ca să fie robust, folosim ordinea: [on_col] + by_cols.
    """
    cols = [on_col] + by_cols
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def evaluate_weekly(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly realized:
      CloseStart = first Close on/after WeekStart (forward)
      CloseEnd   = last  Close on/before TargetDate (backward)
      Realized_Pct = (CloseEnd/CloseStart - 1)*100
    """
    if prices.empty or forecasts.empty:
        return pd.DataFrame()

    forecasts = dedup_forecasts(forecasts)

    # evaluăm doar săptămâni închise
    today = pd.Timestamp(date.today())
    eligible = forecasts[forecasts["TargetDate"] <= today].copy()
    if eligible.empty:
        return pd.DataFrame()

    p = prices[["Date", "Ticker", "Region", "Close"]].copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["Close"] = pd.to_numeric(p["Close"], errors="coerce")
    p = p.dropna(subset=["Date", "Ticker", "Region", "Close"])

    # IMPORTANT: sortare pentru merge_asof (on + by)
    by_cols = ["Ticker", "Region"]

    eligible = _sort_for_asof(eligible, "WeekStart", by_cols)
    p_sorted = _sort_for_asof(p, "Date", by_cols)

    # Start forward merge
    start = pd.merge_asof(
        eligible,
        p_sorted,
        left_on="WeekStart",
        right_on="Date",
        by=by_cols,
        direction="forward",
        suffixes=("", "_px"),
    ).rename(columns={"Date_px": "PriceDateStart", "Close": "CloseStart"})

    # End backward merge
    start = start.dropna(subset=["TargetDate"])
    start = _sort_for_asof(start, "TargetDate", by_cols)

    end = pd.merge_asof(
        start,
        p_sorted,
        left_on="TargetDate",
        right_on="Date",
        by=by_cols,
        direction="backward",
        suffixes=("", "_px2"),
    ).rename(columns={"Date_px2": "PriceDateEnd", "Close": "CloseEnd"})

    df = end.dropna(subset=["CloseStart", "CloseEnd"]).copy()

    df["Realized_Pct"] = (df["CloseEnd"].astype(float) / df["CloseStart"].astype(float) - 1.0) * 100.0
    df["Model_ER_Pct"] = df["ER_Pct"].astype(float)
    df["Error_Pct"] = df["Realized_Pct"] - df["Model_ER_Pct"]
    df["AbsError_Pct"] = df["Error_Pct"].abs()

    model_dir = np.sign(df["Model_ER_Pct"].values)
    real_dir = np.sign(df["Realized_Pct"].values)
    df["DirectionHit"] = (model_dir == real_dir) & (model_dir != 0)

    cols = [
        "WeekStart","TargetDate","Date","ModelVersion","Ticker","Region","HorizonDays",
        "Model_ER_Pct","CloseStart","CloseEnd","Realized_Pct","Error_Pct","AbsError_Pct","DirectionHit","Notes"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[cols].sort_values(["WeekStart", "Region", "Ticker"], kind="mergesort").reset_index(drop=True)


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()

    g = eval_df.groupby(["WeekStart", "ModelVersion", "Region"])
    s = g.agg(
        n=("Ticker","count"),
        hit_rate=("DirectionHit","mean"),
        mean_abs_error=("AbsError_Pct","mean"),
        mean_error=("Error_Pct","mean"),
        median_abs_error=("AbsError_Pct","median"),
    ).reset_index()
    s["hit_rate"] = s["hit_rate"] * 100.0
    return s.sort_values(["WeekStart","Region"], kind="mergesort").reset_index(drop=True)
