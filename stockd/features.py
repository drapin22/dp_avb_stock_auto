from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(close, period=14):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    gain = d.clip(lower=0).tail(period).mean()
    loss = (-d.clip(upper=0)).tail(period).mean()
    if loss == 0: return 100.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))

def _rolling_beta(ar, br, window=60):
    df = pd.concat([ar, br], axis=1).dropna()
    if df.shape[0] < max(10, window//3): return 1.0
    a = df.iloc[-window:,0].astype(float); b = df.iloc[-window:,1].astype(float)
    var = float(np.var(b))
    if var <= 1e-12: return 1.0
    return float(np.cov(a,b)[0,1]) / var

_DIV_CAL = None
_EARN_CAL = None

def _load_div():
    try:
        from stockd import settings
        p = settings.DATA_DIR / "dividend_calendar.csv"
        if p.exists():
            df = pd.read_csv(p); df["ExDivDate"] = pd.to_datetime(df["ExDivDate"], errors="coerce")
            return df.dropna(subset=["ExDivDate"])
    except: pass
    return pd.DataFrame(columns=["Ticker","ExDivDate"])

def _load_earn():
    try:
        from stockd import settings
        p = settings.DATA_DIR / "earnings_calendar.csv"
        if p.exists():
            df = pd.read_csv(p); df["EarningsDate"] = pd.to_datetime(df["EarningsDate"], errors="coerce")
            return df.dropna(subset=["EarningsDate"])
    except: pass
    return pd.DataFrame(columns=["Ticker","EarningsDate"])

def compute_ticker_features(prices_history, as_of=None):
    global _DIV_CAL, _EARN_CAL
    if _DIV_CAL is None: _DIV_CAL = _load_div()
    if _EARN_CAL is None: _EARN_CAL = _load_earn()
    if prices_history is None or prices_history.empty: return pd.DataFrame(columns=["Ticker","Region"])
    as_of = pd.to_datetime(as_of) if as_of is not None else None
    p = prices_history.copy()
    p["Date"]=pd.to_datetime(p["Date"],errors="coerce"); p["Ticker"]=p["Ticker"].astype(str).str.upper().str.strip()
    p["Region"]=p["Region"].astype(str).str.upper().str.strip(); p["Close"]=pd.to_numeric(p["Close"],errors="coerce")
    if as_of is not None: p = p[p["Date"]<=as_of]
    p = p.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])
    if p.empty: return pd.DataFrame(columns=["Ticker","Region"])
    has_vol = "Volume" in p.columns
    proxy = p.groupby(["Region","Date"])["Close"].mean().pct_change().rename("ProxyRet").reset_index()
    proxy["Date"] = pd.to_datetime(proxy["Date"],errors="coerce")
    ro_proxy_ret_5d = np.nan
    rop = proxy[proxy["Region"]=="RO"].sort_values("Date")
    if not rop.empty:
        rr = rop["ProxyRet"].dropna().tail(5)
        if rr.shape[0] >= 2: ro_proxy_ret_5d = (float((1.0+rr).prod())-1.0)*100.0
    reg_ret20 = {}
    for reg, g in p.groupby("Region"):
        vals = []
        for _, tg in g.groupby("Ticker"):
            c = tg.sort_values("Date")["Close"].reset_index(drop=True)
            if len(c) >= 21: vals.append((float(c.iloc[-1])/float(c.iloc[-21])-1.0)*100.0)
        reg_ret20[reg] = float(np.nanmean(vals)) if vals else 0.0
    feats = []
    for (tkr,reg), g in p.groupby(["Ticker","Region"]):
        g = g.sort_values("Date"); close = g["Close"].astype(float).reset_index(drop=True); n = len(close)
        def sr(lb): return (float(close.iloc[-1])/float(close.iloc[-lb-1])-1.0)*100.0 if n>=lb+1 else np.nan
        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std()*np.sqrt(252)*100.0) if len(rets)>=10 else np.nan
        win = close.tail(60) if n>=10 else close
        max_dd = float(((win/win.cummax())-1.0).min()*100.0) if not win.empty else np.nan
        prx = proxy[proxy["Region"]==reg].set_index("Date")["ProxyRet"]
        ar = rets.copy(); ar.index = g["Date"].iloc[1:len(rets)+1].values; ar = ar.reindex(prx.index).dropna()
        beta = _rolling_beta(ar, prx, 60)
        vol_ratio = np.nan
        if has_vol:
            vc = pd.to_numeric(g["Volume"],errors="coerce").astype(float).reset_index(drop=True)
            avg20 = float(vc.tail(20).mean())
            if avg20 > 0: vol_ratio = float(vc.iloc[-1]/avg20)
        ret_20d = sr(20); peer_rel = (ret_20d - reg_ret20.get(reg,0.0)) if not pd.isna(ret_20d) else np.nan
        sma60 = float(close.tail(60).mean()) if n>=20 else float(close.mean())
        regime = 1.0 if float(close.iloc[-1]) > sma60 else 0.0
        rsi_14 = _rsi(close, 14)
        z_price = (float(close.iloc[-1])-float(close.tail(60).mean()))/(float(close.tail(60).std())+1e-8) if n>=20 else 0.0
        days_to_exdiv_norm = post_exdiv_flag = 0.0
        if not _DIV_CAL.empty and as_of is not None:
            td = _DIV_CAL[_DIV_CAL["Ticker"]==tkr].copy()
            if not td.empty:
                td["diff"] = (td["ExDivDate"]-as_of).dt.days
                fut = td[td["diff"]>=0]; past = td[td["diff"]<0]
                if not fut.empty: days_to_exdiv_norm = max(0.0, 1.0-int(fut["diff"].min())/30.0)
                if not past.empty and abs(int(past["diff"].max()))<=5: post_exdiv_flag = 1.0
        earnings_proximity = 0.0
        if not _EARN_CAL.empty and as_of is not None:
            te = _EARN_CAL[_EARN_CAL["Ticker"]==tkr].copy()
            if not te.empty:
                te["diff"] = (te["EarningsDate"]-as_of).dt.days.abs()
                earnings_proximity = max(0.0, 1.0-int(te["diff"].min())/10.0)
        feats.append({"Ticker":tkr,"Region":reg,"ret_5d":sr(5),"ret_20d":ret_20d,"ret_60d":sr(60),"vol_20d":vol_20d,"max_dd_60d":max_dd,"beta":beta,"volume_ratio":vol_ratio,"peer_rel_20d":peer_rel,"regime":regime,"ro_proxy_ret_5d":ro_proxy_ret_5d if reg=="RO" else np.nan,"rsi_14":rsi_14,"z_price_60d":z_price,"days_to_exdiv_norm":days_to_exdiv_norm,"post_exdiv_flag":post_exdiv_flag,"earnings_proximity":earnings_proximity})
    out = pd.DataFrame(feats)
    for c in ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","volume_ratio","peer_rel_20d","regime","ro_proxy_ret_5d","rsi_14","z_price_60d","days_to_exdiv_norm","post_exdiv_flag","earnings_proximity"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
