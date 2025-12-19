from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from stockd import settings


def load_prices(prices_path: Path | None = None) -> pd.DataFrame:
    prices_path = Path(prices_path) if prices_path else settings.PRICES_HISTORY
    if not prices_path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(prices_path)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str)
    df["Region"] = df["Region"].astype(str)
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    return df


def load_prices_history(prices_path: Path | None = None) -> pd.DataFrame:
    return load_prices(prices_path=prices_path)


def load_forecasts(forecasts_path: Path | None = None) -> pd.DataFrame:
    forecasts_path = Path(forecasts_path) if forecasts_path else settings.FORECASTS_STOCKD
    if not forecasts_path.exists():
        return pd.DataFrame(
            columns=["Date", "WeekStart", "TargetDate", "ModelVersion", "Ticker", "Region", "HorizonDays", "ER_Pct", "Notes"]
        )

    df = pd.read_csv(forecasts_path)
    if df.empty:
        return pd.DataFrame(
            columns=["Date", "WeekStart", "TargetDate", "ModelVersion", "Ticker", "Region", "HorizonDays", "ER_Pct", "Notes"]
        )

    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    df["Ticker"] = df["Ticker"].astype(str)
    df["Region"] = df["Region"].astype(str)
    if "ER_Pct" in df.columns:
        df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce")

    df = df.dropna(subset=["WeekStart", "TargetDate", "Ticker", "Region", "ER_Pct"])
    return df


def dedup_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()

    df = forecasts.copy()
    if "Date" in df.columns:
        df = df.sort_values(["WeekStart", "TargetDate", "Ticker", "Region", "Date"])
    else:
        df = df.sort_values(["WeekStart", "TargetDate", "Ticker", "Region"])
    df = df.drop_duplicates(subset=["WeekStart", "TargetDate", "Ticker", "Region"], keep="last")
    return df


def _sorted_for_asof_prices(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    return p.sort_values(["Ticker", "Region", "Date"])


def _sorted_for_asof_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    f = forecasts.copy()
    return f.sort_values(["Ticker", "Region", "WeekStart", "TargetDate"])


def evaluate_weekly(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or forecasts.empty:
        return pd.DataFrame()

    p = _sorted_for_asof_prices(prices)
    f = _sorted_for_asof_forecasts(forecasts)

    start = pd.merge_asof(
        f.sort_values(["Ticker", "Region", "WeekStart"]),
        p,
        left_on="WeekStart",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="forward",
        allow_exact_matches=True,
    ).rename(columns={"Close": "StartClose"})

    end = pd.merge_asof(
        f.sort_values(["Ticker", "Region", "TargetDate"]),
        p,
        left_on="TargetDate",
        right_on="Date",
        by=["Ticker", "Region"],
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"Close": "EndClose"})

    out = f.copy()
    out = out.merge(
        start[["WeekStart", "TargetDate", "Ticker", "Region", "StartClose"]],
        on=["WeekStart", "TargetDate", "Ticker", "Region"],
        how="left",
    )
    out = out.merge(
        end[["WeekStart", "TargetDate", "Ticker", "Region", "EndClose"]],
        on=["WeekStart", "TargetDate", "Ticker", "Region"],
        how="left",
    )

    out["RealizedReturnPct"] = (out["EndClose"] / out["StartClose"] - 1.0) * 100.0
    out["PredictedReturnPct"] = out["ER_Pct"]

    out["hit"] = (
        (out["PredictedReturnPct"] != 0)
        & (out["RealizedReturnPct"] != 0)
        & ((out["PredictedReturnPct"] > 0) == (out["RealizedReturnPct"] > 0))
    )

    out["abs_error_pp"] = (out["PredictedReturnPct"] - out["RealizedReturnPct"]).abs()
    out["bias_pp"] = (out["PredictedReturnPct"] - out["RealizedReturnPct"])

    return out


def summarize(eval_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if eval_df is None or eval_df.empty:
        return {}

    res: Dict[str, Dict[str, float]] = {}
    for region, g in eval_df.groupby("Region"):
        n = float(len(g))
        hit = float(g["hit"].mean()) if n else 0.0
        mae = float(g["abs_error_pp"].mean()) if n else 0.0
        bias = float(g["bias_pp"].mean()) if n else 0.0
        res[str(region)] = {"n": n, "hit": hit, "MAE_pp": mae, "bias_pp": bias}

    return res
