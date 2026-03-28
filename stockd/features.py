from __future__ import annotations
import numpy as np
import pandas as pd


def _rolling_beta(asset_ret, bench_ret, window=60):
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < max(10, window // 3):
        return 1.0
    a = df.iloc[-window:, 0].astype(float)
    b = df.iloc[-window:, 1].astype(float)
    var = float(np.var(b))
    if var <= 1e-12:
        return 1.0
    return float(np.cov(a, b)[0, 1]) / var


def _build_region_proxy_returns(p):
    p2 = p.copy()
    p2["Ret"] = p2.groupby(["Ticker", "Region"])["Close"].pct_change()
    proxy = p2.groupby(["Region", "Date"], as_index=False)["Ret"].mean()
    return proxy.rename(columns={"Ret": "ProxyRet"})


def compute_ticker_features(prices_history, as_of=None):
    """
    Extended feature engineering:
    ret_5d, ret_20d, ret_60d, vol_20d, max_dd_60d, beta,
    volume_ratio (BVB liquidity signal),
    peer_rel_20d (relative strength vs region),
    regime (above/below 60d SMA),
    ro_proxy_ret_5d (legacy)
    """
    if prices_history is None or prices_history.empty:
        return pd.DataFrame(columns=["Ticker","Region","ret_5d","ret_20d","ret_60d",
                                      "vol_20d","max_dd_60d","beta","volume_ratio",
                                      "peer_rel_20d","regime","ro_proxy_ret_5d"])

    as_of = pd.to_datetime(as_of) if as_of is not None else None
    p = prices_history.copy()
    p["Date"]   = pd.to_datetime(p["Date"],  errors="coerce")
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p["Region"] = p["Region"].astype(str).str.upper().str.strip()
    p["Close"]  = pd.to_numeric(p["Close"],  errors="coerce")
    if as_of is not None:
        p = p[p["Date"] <= as_of]
    p = p.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])
    if p.empty:
        return pd.DataFrame(columns=["Ticker","Region"])

    has_volume = "Volume" in p.columns
    proxy = _build_region_proxy_returns(p)
    proxy["Date"] = pd.to_datetime(proxy["Date"], errors="coerce")

    # RO proxy 5d (legacy)
    ro_proxy_ret_5d = np.nan
    ro_proxy = proxy[proxy["Region"] == "RO"].sort_values("Date")
    if not ro_proxy.empty:
        rr = ro_proxy["ProxyRet"].dropna().tail(5)
        if rr.shape[0] >= 2:
            ro_proxy_ret_5d = (float((1.0 + rr).prod()) - 1.0) * 100.0

    # Region avg 20d return for peer_rel
    reg_ret20 = {}
    for reg, g in p.groupby("Region"):
        vals = []
        for _, tg in g.groupby("Ticker"):
            c = tg.sort_values("Date")["Close"].reset_index(drop=True)
            if len(c) >= 21:
                vals.append((float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100.0)
        reg_ret20[reg] = float(np.nanmean(vals)) if vals else 0.0

    feats = []
    for (tkr, reg), g in p.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        close = g["Close"].astype(float).reset_index(drop=True)
        n = len(close)

        def sr(lb):
            return (float(close.iloc[-1]) / float(close.iloc[-lb-1]) - 1.0)*100.0 if n >= lb+1 else np.nan

        ret_5d  = sr(5)
        ret_20d = sr(20)
        ret_60d = sr(60)

        rets = close.pct_change().dropna()
        vol_20d    = float(rets.tail(20).std() * np.sqrt(252) * 100.0) if len(rets) >= 10 else np.nan
        window     = close.tail(60) if n >= 10 else close
        max_dd_60d = float(((window / window.cummax()) - 1.0).min() * 100.0) if not window.empty else np.nan

        prx = proxy[proxy["Region"] == reg].set_index("Date")["ProxyRet"]
        ar  = rets.copy()
        ar.index = g["Date"].iloc[1:len(rets)+1].values
        ar = ar.reindex(prx.index).dropna()
        beta = _rolling_beta(ar, prx, window=60)

        if has_volume:
            vc = pd.to_numeric(g["Volume"], errors="coerce").astype(float).reset_index(drop=True)
            avg20 = float(vc.tail(20).mean())
            volume_ratio = float(vc.iloc[-1] / avg20) if avg20 > 0 else 1.0
        else:
            volume_ratio = np.nan

        peer_rel_20d = (ret_20d - reg_ret20.get(reg, 0.0)) if not np.isnan(ret_20d if ret_20d is not None else float("nan")) else np.nan
        sma60  = float(close.tail(60).mean()) if n >= 20 else float(close.mean())
        regime = 1.0 if float(close.iloc[-1]) > sma60 else 0.0

        feats.append({
            "Ticker": tkr, "Region": reg,
            "ret_5d": ret_5d, "ret_20d": ret_20d, "ret_60d": ret_60d,
            "vol_20d": vol_20d, "max_dd_60d": max_dd_60d, "beta": beta,
            "volume_ratio": volume_ratio, "peer_rel_20d": peer_rel_20d, "regime": regime,
            "ro_proxy_ret_5d": ro_proxy_ret_5d if reg == "RO" else np.nan,
        })

    out = pd.DataFrame(feats)
    for c in ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta",
              "volume_ratio","peer_rel_20d","regime","ro_proxy_ret_5d"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
