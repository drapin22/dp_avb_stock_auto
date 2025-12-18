# stockd/volatility.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stockd import settings


@dataclass
class VolConfig:
    lookback_days: int = 60
    horizon_days: int = 5
    min_scale: float = 0.35
    max_scale: float = 1.00
    fallback_target_weekly_vol: float = 3.0  # in percent


def _prices_path() -> Path:
    # prefer settings.PRICES_HISTORY if exists
    p = getattr(settings, "PRICES_HISTORY", None)
    if p:
        return Path(p)
    return settings.DATA_DIR / "prices_history.csv"


def compute_weekly_vol_metrics(cfg: Optional[VolConfig] = None) -> pd.DataFrame:
    """
    Returns dataframe with:
      Date (latest), Ticker, Region, vol_weekly_pct, target_weekly_pct, vol_scale
    vol_weekly_pct is realized weekly vol based on daily returns.
    vol_scale = clamp(target / vol_weekly, min_scale..max_scale)
    """
    cfg = cfg or VolConfig()
    path = _prices_path()
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "vol_weekly_pct", "target_weekly_pct", "vol_scale"])

    df = pd.read_csv(path)
    needed = {"Date", "Ticker", "Region", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "vol_weekly_pct", "target_weekly_pct", "vol_scale"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"]).copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()

    df = df.sort_values(["Ticker", "Date"])
    df["ret_1d"] = df.groupby("Ticker")["Close"].pct_change()

    # last lookback per ticker
    def last_n(g: pd.DataFrame) -> pd.DataFrame:
        return g.tail(cfg.lookback_days)

    df_lb = df.groupby("Ticker", group_keys=False).apply(last_n)

    # realized weekly vol (percent) = std(daily returns) * sqrt(horizon) * 100
    agg = (
        df_lb.groupby(["Ticker", "Region"])
        .agg(
            last_date=("Date", "max"),
            std_1d=("ret_1d", "std"),
            n=("ret_1d", "count"),
        )
        .reset_index()
    )

    agg["std_1d"] = agg["std_1d"].fillna(0.0)
    agg["vol_weekly_pct"] = agg["std_1d"] * np.sqrt(cfg.horizon_days) * 100.0
    agg["vol_weekly_pct"] = agg["vol_weekly_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # region target: median vol, with fallback
    targets = (
        agg[agg["vol_weekly_pct"] > 0]
        .groupby("Region")["vol_weekly_pct"]
        .median()
        .to_dict()
    )

    def target_for_region(r: str) -> float:
        t = targets.get(r)
        if t is None or not np.isfinite(t) or t <= 0:
            return float(cfg.fallback_target_weekly_vol)
        # slight conservatism
        return float(max(1.5, min(6.0, t * 0.85)))

    agg["target_weekly_pct"] = agg["Region"].astype(str).apply(target_for_region)

    def scale(row) -> float:
        v = float(row["vol_weekly_pct"])
        t = float(row["target_weekly_pct"])
        if v <= 0.25:
            return 1.0
        s = t / v
        s = max(cfg.min_scale, min(cfg.max_scale, s))
        return float(s)

    agg["vol_scale"] = agg.apply(scale, axis=1)
    agg["Date"] = agg["last_date"].dt.date.astype(str)

    out = agg[["Date", "Ticker", "Region", "vol_weekly_pct", "target_weekly_pct", "vol_scale"]].copy()
    out = out.sort_values(["Region", "Ticker"]).reset_index(drop=True)
    return out


def save_vol_metrics() -> Path:
    df = compute_weekly_vol_metrics()
    out_path = settings.DATA_DIR / "volatility_metrics.csv"
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
