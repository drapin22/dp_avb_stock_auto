# stockd/evaluation.py
from __future__ import annotations

from datetime import date
import pandas as pd
import numpy as np

from stockd import settings


def _load_prices() -> pd.DataFrame:
    # preferă prices_all.csv dacă există, altfel prices_history.csv
    path = settings.PRICES_ALL_CSV if settings.PRICES_ALL_CSV.exists() else settings.PRICES_HISTORY
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["Ticker"] = df.get("Ticker", "").astype(str)
    df["Region"] = df.get("Region", "").astype(str)
    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    df = df[df["Close"] > 0]

    # sanity: elimină outlier jumps foarte mari (de obicei scrape glitch)
    df = df.sort_values(["Ticker", "Region", "Date"]).copy()
    df["pct"] = df.groupby(["Ticker", "Region"])["Close"].pct_change()
    df = df[(df["pct"].isna()) | (df["pct"].abs() <= 0.60)]
    df = df.drop(columns=["pct"], errors="ignore")

    return df


def _load_forecasts() -> pd.DataFrame:
    path = settings.FORECASTS_FILE
    if not path.exists():
        return pd.DataFrame(columns=[
            "Date","WeekStart","TargetDate","ModelVersion","Ticker","Region","HorizonDays","ER_Pct","Notes"
        ])

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["WeekStart"] = pd.to_datetime(df.get("WeekStart"), errors="coerce")
    df["TargetDate"] = pd.to_datetime(df.get("TargetDate"), errors="coerce")
    df["Ticker"] = df.get("Ticker", "").astype(str)
    df["Region"] = df.get("Region", "").astype(str)
    df["ModelVersion"] = df.get("ModelVersion", "").astype(str)
    df["ER_Pct"] = pd.to_numeric(df.get("ER_Pct"), errors="coerce")

    df = df.dropna(subset=["Date", "WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"])
    df = df.sort_values("Date")
    df = (
        df.groupby(["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"], as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )
    return df


def evaluate_weekly(prices: pd.DataFrame | None = None, forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    prices = _load_prices() if prices is None else prices.copy()
    forecasts = _load_forecasts() if forecasts is None else forecasts.copy()

    if prices.empty or forecasts.empty:
        return pd.DataFrame()

    today_ts = pd.Timestamp(date.today())
    eligible = forecasts[forecasts["TargetDate"] < today_ts].copy()
    if eligible.empty:
        return pd.DataFrame()

    # pregătire merge_asof (obligatoriu sort by + on)
    p = prices[["Date", "Ticker", "Region", "Close"]].copy()
    p = p.sort_values(["Ticker", "Region", "Date"])

    eligible = eligible.sort_values(["Ticker", "Region", "WeekStart"]).copy()
    start = pd.merge_asof(
        eligible,
        p,
        left_on="WeekStart",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="backward",
        suffixes=("", "_px"),
    ).rename(columns={"Date_px": "PriceDateStart", "Close": "CloseStart"})

    start = start.sort_values(["Ticker", "Region", "TargetDate"]).copy()
    end = pd.merge_asof(
        start,
        p,
        left_on="TargetDate",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="backward",
        suffixes=("", "_px2"),
    ).rename(columns={"Date_px2": "PriceDateEnd", "Close": "CloseEnd"})

    df = end.copy()
    df["Model_ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce")
    df["Realized_Pct"] = (df["CloseEnd"] / df["CloseStart"] - 1.0) * 100.0

    df["Error_Pct"] = df["Realized_Pct"] - df["Model_ER_Pct"]
    df["AbsError_Pct"] = df["Error_Pct"].abs()

    # direcție: HIT dacă semn identic și ambele non-zero
    df["DirectionHit"] = 0
    mask = (df["Realized_Pct"].abs() > 1e-12) & (df["Model_ER_Pct"].abs() > 1e-12)
    df.loc[mask, "DirectionHit"] = (np.sign(df.loc[mask, "Realized_Pct"]) == np.sign(df.loc[mask, "Model_ER_Pct"])).astype(int)

    out_cols = [
        "WeekStart","TargetDate","Date","ModelVersion","Ticker","Region","HorizonDays",
        "Model_ER_Pct","CloseStart","CloseEnd","Realized_Pct","Error_Pct","AbsError_Pct","DirectionHit","Notes"
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[out_cols].copy()


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["WeekStart","ModelVersion","Region","n","hit_rate","mean_abs_error","mean_error","median_abs_error"])

    g = eval_df.copy()
    g["AbsError_Pct"] = pd.to_numeric(g["AbsError_Pct"], errors="coerce")
    g["Error_Pct"] = pd.to_numeric(g["Error_Pct"], errors="coerce")
    g["DirectionHit"] = pd.to_numeric(g["DirectionHit"], errors="coerce")

    grp = g.groupby(["WeekStart","ModelVersion","Region"], dropna=False)
    res = grp.agg(
        n=("Ticker","count"),
        hit_rate=("DirectionHit","mean"),
        mean_abs_error=("AbsError_Pct","mean"),
        mean_error=("Error_Pct","mean"),
        median_abs_error=("AbsError_Pct","median"),
    ).reset_index()

    return res
