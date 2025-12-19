from __future__ import annotations

"""
Feature engineering for StockD.

Key fix:
- compute real betas using *region proxy returns* built from prices_history,
  so we do NOT need extra benchmark downloads.
"""

import pandas as pd
import numpy as np


def _safe_pct_change(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.pct_change()


def _rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 60) -> float:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < max(10, window // 3):
        return 1.0
    a = df.iloc[-window:, 0].astype(float)
    b = df.iloc[-window:, 1].astype(float)
    var = float(np.var(b))
    if var <= 1e-12:
        return 1.0
    cov = float(np.cov(a, b)[0, 1])
    return cov / var


def _build_region_proxy_returns(prices_history: pd.DataFrame) -> pd.DataFrame:
    p = prices_history.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p["Region"] = p["Region"].astype(str).str.upper().str.strip()
    p["Close"] = pd.to_numeric(p["Close"], errors="coerce")
    p = p.dropna(subset=["Date", "Ticker", "Region", "Close"]).sort_values(["Ticker", "Region", "Date"])

    p["Ret"] = p.groupby(["Ticker", "Region"])["Close"].pct_change()
    proxy = p.groupby(["Region", "Date"], as_index=False)["Ret"].mean()
    return proxy.rename(columns={"Ret": "ProxyRet"})


def compute_ticker_features(prices_history: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if prices_history is None or prices_history.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ret_20d", "ret_60d", "vol_20d", "max_dd_60d", "beta", "ro_proxy_ret_5d"])

    as_of = pd.to_datetime(as_of) if as_of is not None else None

    p = prices_history.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p["Region"] = p["Region"].astype(str).str.upper().str.strip()
    p["Close"] = pd.to_numeric(p["Close"], errors="coerce")
    p = p.dropna(subset=["Date", "Ticker", "Region", "Close"]).sort_values(["Ticker", "Region", "Date"])

    if as_of is not None:
        p = p[p["Date"] <= as_of]

    if p.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ret_20d", "ret_60d", "vol_20d", "max_dd_60d", "beta", "ro_proxy_ret_5d"])

    proxy = _build_region_proxy_returns(p)
    proxy["Date"] = pd.to_datetime(proxy["Date"], errors="coerce")

    # proxy 5d return for RO (helpful when BENCH isn't available)
    ro_proxy_ret_5d = np.nan
    ro_proxy = proxy[proxy["Region"] == "RO"].sort_values("Date")
    if not ro_proxy.empty:
        rr = ro_proxy["ProxyRet"].dropna().tail(5)
        if rr.shape[0] >= 2:
            ro_proxy_ret_5d = (float((1.0 + rr).prod()) - 1.0) * 100.0

    feats = []
    for (tkr, reg), g in p.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        close = g["Close"].astype(float)

        # returns
        ret_20d = (close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0 if close.shape[0] >= 21 else np.nan
        ret_60d = (close.iloc[-1] / close.iloc[-61] - 1.0) * 100.0 if close.shape[0] >= 61 else np.nan

        # vol on daily returns
        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std() * np.sqrt(252) * 100.0) if rets.shape[0] >= 10 else np.nan

        # max drawdown over last 60 obs
        window = close.tail(60) if close.shape[0] >= 10 else close
        run_max = window.cummax()
        dd = (window / run_max) - 1.0
        max_dd_60d = float(dd.min() * 100.0) if dd.shape[0] else np.nan

        # beta vs region proxy
        proxy_reg = proxy[proxy["Region"] == reg].set_index("Date")["ProxyRet"]
        asset_ret = rets
        asset_ret.index = g["Date"].iloc[1:len(rets) + 1].values  # align index with dates after pct_change
        asset_ret = asset_ret.reindex(proxy_reg.index).dropna()

        beta = _rolling_beta(asset_ret, proxy_reg, window=60)

        feats.append(
            {
                "Ticker": tkr,
                "Region": reg,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "vol_20d": vol_20d,
                "max_dd_60d": max_dd_60d,
                "beta": beta,
                "ro_proxy_ret_5d": ro_proxy_ret_5d if reg == "RO" else np.nan,
            }
        )

    out = pd.DataFrame(feats)
    # cleanup
    for c in ["ret_20d", "ret_60d", "vol_20d", "max_dd_60d", "beta", "ro_proxy_ret_5d"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
