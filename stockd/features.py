from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _pct_return(close: pd.Series, n: int) -> float:
    if close.shape[0] < n + 1:
        return float("nan")
    a = float(close.iloc[-(n + 1)])
    b = float(close.iloc[-1])
    if a == 0:
        return float("nan")
    return (b / a - 1.0) * 100.0


def _max_drawdown_pct(close: pd.Series, n: int) -> float:
    s = close.tail(n).astype(float)
    if s.empty:
        return float("nan")
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return float(dd.min() * 100.0)


def build_ro_proxy_returns(prices_history: pd.DataFrame) -> pd.Series:
    df = prices_history.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    ro = df[df["Region"] == "RO"].copy()
    if ro.empty:
        return pd.Series(dtype=float)

    ro = ro.sort_values(["Ticker", "Date"])
    ro["ret"] = ro.groupby("Ticker")["Close"].pct_change()
    proxy = ro.groupby("Date")["ret"].mean().dropna().sort_index()
    return proxy


def rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 60) -> float:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < max(20, window // 2):
        return 1.0
    df = df.iloc[-window:]
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    vx = np.var(x)
    if vx <= 1e-12:
        return 1.0
    return float(np.cov(y, x)[0, 1] / vx)


def compute_ticker_features(prices_history: pd.DataFrame, holdings: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Returnează:
      - features per (Ticker, Region)
      - debug dict cu proxy RO returns
    """
    if prices_history is None or prices_history.empty or holdings is None or holdings.empty:
        cols = ["Ticker","Region","last_close","ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","ro_proxy_ret_5d"]
        return pd.DataFrame(columns=cols), {"ro_proxy": "EMPTY"}

    df = prices_history.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    wanted = holdings[["Ticker","Region"]].copy()
    wanted["Ticker"] = wanted["Ticker"].astype(str).str.upper().str.strip()
    wanted["Region"] = wanted["Region"].astype(str).str.upper().str.strip()
    wanted = wanted.drop_duplicates()

    df = df.merge(wanted, on=["Ticker","Region"], how="inner")
    if df.empty:
        cols = ["Ticker","Region","last_close","ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","ro_proxy_ret_5d"]
        return pd.DataFrame(columns=cols), {"ro_proxy": "NO_MATCH"}

    # ro proxy
    ro_proxy = build_ro_proxy_returns(prices_history)
    ro_proxy_ret_5d = 0.0
    if not ro_proxy.empty and ro_proxy.shape[0] >= 6:
        ro_proxy_ret_5d = float((ro_proxy.iloc[-1] / ro_proxy.iloc[-6] - 1.0) * 100.0)

    # compute per ticker
    feats = []
    df = df.sort_values(["Ticker","Region","Date"])
    for (t, r), g in df.groupby(["Ticker","Region"]):
        close = g["Close"].astype(float).reset_index(drop=True)
        last_close = float(close.iloc[-1])

        ret5 = _pct_return(close, 5)
        ret20 = _pct_return(close, 20)
        ret60 = _pct_return(close, 60)

        if close.shape[0] >= 21:
            daily = close.pct_change()
            vol20 = float(daily.tail(20).std() * (20 ** 0.5) * 100.0)
        else:
            vol20 = float("nan")

        mdd60 = _max_drawdown_pct(close, 60)

        feats.append({
            "Ticker": t,
            "Region": r,
            "last_close": last_close,
            "ret_5d": ret5,
            "ret_20d": ret20,
            "ret_60d": ret60,
            "vol_20d": vol20,
            "max_dd_60d": mdd60,
            "beta": 1.0,  # se setează în engine (cu benchmark returns)
            "ro_proxy_ret_5d": ro_proxy_ret_5d if r == "RO" else 0.0
        })

    fdf = pd.DataFrame(feats)
    return fdf, {"ro_proxy_ret_5d": ro_proxy_ret_5d}
