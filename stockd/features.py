from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(close, period=14):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    gains = d.clip(lower=0).rolling(period).mean()
    losses = (-d.clip(upper=0)).rolling(period).mean()
    rs = gains.iloc[-1] / losses.iloc[-1] if losses.iloc[-1] > 1e-10 else 100.0
    return float(100.0 - 100.0 / (1.0 + rs))

def _z_price(close, window=60):
    if len(close) < window: return 0.0
    w = close.tail(window); mu, sigma = float(w.mean()), float(w.std())
    return float((close.iloc[-1] - mu) / sigma) if sigma > 1e-10 else 0.0

def _load_div_cal():
    try:
        from stockd import settings
        p = settings.DATA_DIR / "dividend_calendar.csv"
        if not p.exists(): return None
        df = pd.read_csv(p)
        df["ExDivDate"] = pd.to_datetime(df["ExDivDate"], errors="coerce")
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        return df.dropna(subset=["ExDivDate"])
    except: return None

def _rolling_beta(a, b, window=60):
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < max(10, window // 3): return 1.0
    a2, b2 = df.iloc[-window:, 0].astype(float), df.iloc[-window:, 1].astype(float)
    var = float(np.var(b2))
    if var <= 1e-12: return 1.0
    return float(np.cov(a2, b2)[0, 1]) / var

def _proxy_returns(p):
    p2 = p.copy()
    p2["Ret"] = p2.groupby(["Ticker", "Region"])["Close"].pct_change()
    return p2.groupby(["Region", "Date"], as_index=False)["Ret"].mean().rename(columns={"Ret": "ProxyRet"})

def compute_ticker_features(prices_history, as_of=None):
    if prices_history is None or prices_history.empty:
        return pd.DataFrame(columns=["Ticker", "Region"])
    as_of = pd.to_datetime(as_of) if as_of is not None else None
    p = prices_history.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p["Region"] = p["Region"].astype(str).str.upper().str.strip()
    p["Close"] = pd.to_numeric(p["Close"], errors="coerce")
    if as_of is not None: p = p[p["Date"] <= as_of]
    p = p.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])
    if p.empty: return pd.DataFrame(columns=["Ticker","Region"])

    has_volume = "Volume" in p.columns
    proxy = _proxy_returns(p)
    proxy["Date"] = pd.to_datetime(proxy["Date"], errors="coerce")

    ro_proxy_ret_5d = np.nan
    rp = proxy[proxy["Region"] == "RO"].sort_values("Date")
    if not rp.empty:
        rr = rp["ProxyRet"].dropna().tail(5)
        if rr.shape[0] >= 2: ro_proxy_ret_5d = (float((1.0 + rr).prod()) - 1.0) * 100.0

    reg_ret20 = {}
    for reg, g in p.groupby("Region"):
        vals = []
        for _, tg in g.groupby("Ticker"):
            c = tg.sort_values("Date")["Close"].reset_index(drop=True)
            if len(c) >= 21: vals.append((float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100.0)
        reg_ret20[reg] = float(np.nanmean(vals)) if vals else 0.0

    div_cal = _load_div_cal()
    feats = []
    for (tkr, reg), g in p.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        close = g["Close"].astype(float).reset_index(drop=True)
        n = len(close)

        def sr(lb): return (float(close.iloc[-1])/float(close.iloc[-lb-1])-1.0)*100.0 if n >= lb+1 else np.nan
        ret_5d = sr(5); ret_20d = sr(20); ret_60d = sr(60)

        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std() * np.sqrt(252) * 100.0) if len(rets) >= 10 else np.nan
        window = close.tail(60) if n >= 10 else close
        max_dd = float(((window / window.cummax()) - 1.0).min() * 100.0) if not window.empty else np.nan

        prx = proxy[proxy["Region"] == reg].set_index("Date")["ProxyRet"]
        ar = rets.copy(); ar.index = g["Date"].iloc[1:len(rets)+1].values
        ar = ar.reindex(prx.index).dropna()
        beta = _rolling_beta(ar, prx, 60)

        volume_ratio = np.nan
        if has_volume:
            vc = pd.to_numeric(g["Volume"], errors="coerce").astype(float).reset_index(drop=True)
            avg20 = float(vc.tail(20).mean())
            if avg20 > 0: volume_ratio = float(vc.iloc[-1] / avg20)

        peer_rel = (ret_20d - reg_ret20.get(reg, 0.0)) if ret_20d is not None and not np.isnan(float(ret_20d) if ret_20d is not None else float("nan")) else np.nan
        sma60 = float(close.tail(60).mean()) if n >= 20 else float(close.mean())
        regime = 1.0 if float(close.iloc[-1]) > sma60 else 0.0
        rsi_14 = _rsi(close, 14)
        z_price_60d = _z_price(close, 60)

        days_to_exdiv_norm = 0.0
        post_exdiv_flag = 0.0
        if div_cal is not None and as_of is not None:
            td = div_cal[div_cal["Ticker"] == tkr]
            if not td.empty:
                fut = td[td["ExDivDate"] >= as_of].sort_values("ExDivDate")
                past = td[td["ExDivDate"] < as_of].sort_values("ExDivDate")
                if not fut.empty:
                    days = (fut.iloc[0]["ExDivDate"] - as_of).days
                    days_to_exdiv_norm = float(max(0, 30 - days) / 30.0)
                if not past.empty:
                    days_past = (as_of - past.iloc[-1]["ExDivDate"]).days
                    post_exdiv_flag = 1.0 if days_past <= 5 else 0.0

        feats.append({"Ticker": tkr, "Region": reg,
            "ret_5d": ret_5d, "ret_20d": ret_20d, "ret_60d": ret_60d,
            "vol_20d": vol_20d, "max_dd_60d": max_dd, "beta": beta,
            "volume_ratio": volume_ratio, "peer_rel_20d": peer_rel, "regime": regime,
            "rsi_14": rsi_14, "z_price_60d": z_price_60d,
            "days_to_exdiv_norm": days_to_exdiv_norm, "post_exdiv_flag": post_exdiv_flag,
            "ro_proxy_ret_5d": ro_proxy_ret_5d if reg == "RO" else np.nan})

    out = pd.DataFrame(feats)
    for c in ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta",
              "volume_ratio","peer_rel_20d","regime","rsi_14","z_price_60d",
              "days_to_exdiv_norm","post_exdiv_flag","ro_proxy_ret_5d"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
