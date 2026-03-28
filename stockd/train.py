"""
stockd/train.py v2
- 3 separate models: RO, EU, US with region-specific features
- RSI-14, z_price_60d, dividend calendar (days_to_exdiv_norm, post_exdiv_flag)
- XGBoost vs Ridge — keeps whichever wins on validation MAE
- Saves model_state_RO.json, model_state_EU.json, model_state_US.json
"""
from __future__ import annotations
import json, logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from stockd import settings

log = logging.getLogger(__name__)

FEATURES_RO = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta",
               "volume_ratio","peer_rel_20d","regime","rsi_14","z_price_60d",
               "days_to_exdiv_norm","post_exdiv_flag","bench_ret_5d"]
FEATURES_US = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta",
               "volume_ratio","peer_rel_20d","regime","rsi_14","z_price_60d",
               "vix_level","dxy_ret_5d","bench_ret_5d"]
FEATURES_EU = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta",
               "volume_ratio","peer_rel_20d","regime","rsi_14","z_price_60d",
               "bench_ret_5d"]
REGION_FEATURES = {"RO":FEATURES_RO,"US":FEATURES_US,"EU":FEATURES_EU}
ALL_FEATURES = sorted(set(f for fs in REGION_FEATURES.values() for f in fs))
MIN_TRAIN_WEEKS = 8
RIDGE_ALPHA = 0.3

def _rsi(close, p=14):
    d=close.diff().dropna()
    if len(d)<p: return 50.0
    g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
    rs=g.iloc[-1]/l.iloc[-1] if l.iloc[-1]>1e-10 else 100.0
    return float(100.0-100.0/(1.0+rs))

def _zp(close, w=60):
    if len(close)<w: return 0.0
    win=close.tail(w); mu,sig=float(win.mean()),float(win.std())
    return float((close.iloc[-1]-mu)/sig) if sig>1e-10 else 0.0

def load_div_cal():
    p=settings.DATA_DIR/"dividend_calendar.csv"
    if not p.exists(): return pd.DataFrame(columns=["Ticker","ExDivDate"])
    df=pd.read_csv(p); df["ExDivDate"]=pd.to_datetime(df["ExDivDate"],errors="coerce")
    df["Ticker"]=df["Ticker"].astype(str).str.upper().str.strip()
    return df.dropna(subset=["ExDivDate"])

def feats_as_of(prices, as_of, div_cal=None):
    p=prices[prices["Date"]<=as_of].copy()
    if p.empty: return pd.DataFrame(columns=["Ticker","Region"]+ALL_FEATURES)
    proxy=p.groupby(["Region","Date"])["Close"].mean().pct_change().rename("ProxyRet").reset_index()
    rr20={}
    for reg,g in p.groupby("Region"):
        vals=[]; [(vals.append((float(c.iloc[-1])/float(c.iloc[-21])-1.0)*100.0)) for _,tg in g.groupby("Ticker") for c in [tg.sort_values("Date")["Close"].reset_index(drop=True)] if len(c)>=21]
        rr20[reg]=float(np.nanmean(vals)) if vals else 0.0
    hv="Volume" in p.columns; rows=[]
    for (tkr,reg),g in p.groupby(["Ticker","Region"]):
        g=g.sort_values("Date"); cl=g["Close"].astype(float).reset_index(drop=True); n=len(cl)
        def sr(lb): return (float(cl.iloc[-1])/float(cl.iloc[-lb-1])-1.0)*100.0 if n>=lb+1 else 0.0
        rt=cl.pct_change().dropna()
        v20=float(rt.tail(20).std()*np.sqrt(252)*100.0) if len(rt)>=10 else 20.0
        win=cl.tail(60) if n>=10 else cl
        mdd=float(((win/win.cummax())-1.0).min()*100.0)
        prx=proxy[proxy["Region"]==reg].set_index("Date")["ProxyRet"]
        ar=rt.copy(); ar.index=g["Date"].iloc[1:len(rt)+1].values; ar=ar.reindex(prx.index).dropna()
        if len(ar)>=10:
            vb=float(prx.reindex(ar.index).var()); beta=float(ar.cov(prx.reindex(ar.index))/vb) if vb>1e-12 else 1.0
        else: beta=1.0
        vr=1.0
        if hv:
            vc=pd.to_numeric(g["Volume"],errors="coerce").astype(float).reset_index(drop=True); a20=float(vc.tail(20).mean())
            if a20>0: vr=float(vc.iloc[-1]/a20)
        r20=sr(20); pr=r20-rr20.get(reg,0.0)
        s60=float(cl.tail(60).mean()) if n>=20 else float(cl.mean())
        rgm=1.0 if float(cl.iloc[-1])>s60 else 0.0
        rsi=_rsi(cl,14); zp=_zp(cl,60)
        de=pef=0.0
        if div_cal is not None and not div_cal.empty:
            td=div_cal[div_cal["Ticker"]==tkr]
            if not td.empty:
                fut=td[td["ExDivDate"]>=as_of].sort_values("ExDivDate"); pst=td[td["ExDivDate"]<as_of].sort_values("ExDivDate")
                if not fut.empty: days=(fut.iloc[0]["ExDivDate"]-as_of).days; de=float(max(0,30-days)/30.0)
                if not pst.empty: dp=(as_of-pst.iloc[-1]["ExDivDate"]).days; pef=1.0 if dp<=5 else 0.0
        rows.append({"Ticker":tkr,"Region":reg,"ret_5d":sr(5),"ret_20d":r20,"ret_60d":sr(60),
            "vol_20d":v20,"max_dd_60d":mdd,"beta":beta,"volume_ratio":vr,"peer_rel_20d":pr,"regime":rgm,
            "rsi_14":rsi,"z_price_60d":zp,"days_to_exdiv_norm":de,"post_exdiv_flag":pef,
            "vix_level":0.0,"dxy_ret_5d":0.0,"bench_ret_5d":0.0})
    return pd.DataFrame(rows)

def fwd_ret(prices, as_of, horizon=5):
    rows=[]
    for (t,r),g in prices.groupby(["Ticker","Region"]):
        g=g.sort_values("Date"); pa=g[g["Date"]<=as_of]; fu=g[g["Date"]>as_of].head(horizon)
        if pa.empty or fu.empty: continue
        p0,p1=float(pa.iloc[-1]["Close"]),float(fu.iloc[-1]["Close"])
        if p0>0: rows.append({"Ticker":t,"Region":r,"fwd_ret":(p1/p0-1.0)*100.0})
    return pd.DataFrame(rows)

def ridge_fit(X, y, a=RIDGE_ALPHA):
    A=X.T@X+a*np.eye(X.shape[1])
    try: c=np.linalg.solve(A,X.T@y)
    except np.linalg.LinAlgError: c=np.zeros(X.shape[1])
    return c,float(np.mean(y)-X.mean(axis=0)@c)

def fit_best(Xtr,ytr,Xv,yv,cols):
    c,ic=ridge_fit(Xtr,ytr); rm=float(np.mean(np.abs(ic+Xv@c-yv)))
    xm=float("inf"); xg=None
    try:
        from xgboost import XGBRegressor
        xg=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
        xg.fit(Xtr,ytr); xm=float(np.mean(np.abs(xg.predict(Xv)-yv)))
        if xm<rm*0.95: log.info(f"XGB wins {xm:.4f} vs Ridge {rm:.4f}")
        else: xg=None
    except ImportError: pass
    if xg: return {"type":"xgboost","coef":c.tolist(),"intercept":ic},lambda X:xg.predict(X),xm
    return {"type":"ridge","coef":c.tolist(),"intercept":ic},lambda X,c=c,ic=ic:ic+X@c,rm

def walkforward_region(prices, region, fc, div_cal=None, horizon=5):
    rp=prices[prices["Region"]==region].copy()
    mons=sorted(set(d for d in prices["Date"].unique() if pd.Timestamp(d).dayofweek==0))
    td=[]; er=[]; cp=lambda X:np.full(X.shape[0],0.1)
    for i,m in enumerate(mons[:-1]):
        ts=pd.Timestamp(m)
        fts=feats_as_of(prices,ts,div_cal); fts=fts[fts["Region"]==region]
        fw=fwd_ret(rp,ts,horizon); mg=fts.merge(fw,on=["Ticker","Region"],how="left")
        if i>=MIN_TRAIN_WEEKS and td:
            tdf=pd.concat(td,ignore_index=True).dropna(subset=["fwd_ret"]+fc)
            if len(tdf)>=8:
                Xa=tdf[fc].fillna(0.0).values; ya=tdf["fwd_ret"].values
                sp=max(4,int(len(Xa)*0.8)); Xtr,ytr=Xa[:sp],ya[:sp]; Xv,yv=Xa[sp:],ya[sp:]
                if len(Xv)>=2: _,cp,_=fit_best(Xtr,ytr,Xv,yv,fc)
                else: c2,ic2=ridge_fit(Xa,ya); cp=lambda X,c=c2,ic=ic2:ic+X@c
            Xp=mg[fc].fillna(0.0).values; ps=cp(Xp)
            for j,(_,row) in enumerate(mg.iterrows()):
                pred=float(ps[j]); actual=float(row["fwd_ret"]) if not pd.isna(row.get("fwd_ret")) else float("nan")
                err=(pred-actual) if not np.isnan(actual) else float("nan")
                er.append({"WeekStart":str(ts.date()),"Ticker":row["Ticker"],"Region":row["Region"],
                    "Predicted_ER":round(pred,4),"Actual_ER":round(actual,4) if not np.isnan(actual) else None,
                    "Error_Pct":round(err,4) if not np.isnan(err) else None,
                    "AbsError_Pct":round(abs(err),4) if not np.isnan(err) else None,
                    "DirectionHit":1.0 if pred*actual>0 else(0.0 if not np.isnan(actual) else None)})
        td.append(mg)
    adf=pd.concat(td,ignore_index=True).dropna(subset=["fwd_ret"]+fc) if td else pd.DataFrame()
    st={"feature_cols":fc,"coef":[0.0]*len(fc),"intercept":0.1,"type":"ridge","region":region,"n_samples":0}
    if len(adf)>=4:
        Xa=adf[fc].fillna(0.0).values; ya=adf["fwd_ret"].values
        sp=max(4,int(len(Xa)*0.8)); Xtr,ytr=Xa[:sp],ya[:sp]; Xv,yv=Xa[sp:],ya[sp:]
        sd,_,_ = fit_best(Xtr,ytr,Xv,yv,fc) if len(Xv)>=2 else ({**dict(zip(["coef","intercept"],list(ridge_fit(Xa,ya))[::-1])),"type":"ridge"},None,None)
        if sd: st.update(sd)
        st["n_samples"]=len(adf); st["region"]=region; st["feature_cols"]=fc
        lm=mons[-2] if len(mons)>=2 else mons[-1]; st["trained_on"]=str(pd.Timestamp(lm).date())
        ca=np.array(st["coef"]); st["top_features"]={k:round(v,4) for k,v in sorted(zip(fc,ca),key=lambda x:abs(x[1]),reverse=True)[:6]}
    return pd.DataFrame(er), st

def retrain_and_save(prices_path=None):
    from stockd.scoring import compute_scores, save_scores
    pp=prices_path or settings.PRICES_HISTORY
    if not pp.exists(): return {"error":f"Not found: {pp}"}
    prices=pd.read_csv(pp)
    prices["Date"]=pd.to_datetime(prices["Date"],errors="coerce")
    prices["Close"]=pd.to_numeric(prices["Close"],errors="coerce")
    prices["Ticker"]=prices["Ticker"].astype(str).str.upper().str.strip()
    prices["Region"]=prices["Region"].astype(str).str.upper().str.strip()
    prices=prices.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])
    dc=load_div_cal(); log.info(f"Div cal: {len(dc)} entries")
    summary={"regions":{}}; ae=[]
    for reg in ["RO","EU","US"]:
        fc=REGION_FEATURES[reg]; log.info(f"Training {reg} ({len(fc)} features)...")
        edf,st=walkforward_region(prices,reg,fc,dc)
        mp=settings.DATA_DIR/f"model_state_{reg}.json"; mp.write_text(json.dumps(st,indent=2,ensure_ascii=False))
        log.info(f"Saved model_state_{reg}.json type={st.get('type')} n={st.get('n_samples')}")
        if not edf.empty:
            ae.append(edf); h=edf["DirectionHit"].dropna(); m=edf["AbsError_Pct"].dropna().mean()
            summary["regions"][reg]={"type":st.get("type","ridge"),"n":st.get("n_samples",0),
                "hit_rate":round(float(h.mean())*100,1),"mae":round(float(m),3),"top":st.get("top_features",{})}
    ce=pd.concat(ae,ignore_index=True) if ae else pd.DataFrame()
    if not ce.empty:
        ep=settings.DATA_DIR/"backtest_eval.csv"
        if ep.exists():
            ex=pd.read_csv(ep); ce=pd.concat([ex,ce],ignore_index=True)
            ce.drop_duplicates(subset=["WeekStart","Ticker","Region"],keep="last",inplace=True)
        ce.to_csv(ep,index=False)
        sc=compute_scores(ce)
        if not sc.empty: save_scores(sc); summary["n_scored"]=len(sc)
    (settings.DATA_DIR/"model_state.json").write_text(json.dumps({
        "feature_cols":ALL_FEATURES,"coef":[0.0]*len(ALL_FEATURES),"intercept":0.1,
        "type":"multi_region","regions":["RO","EU","US"],
        "n_samples":sum(v.get("n",0) for v in summary["regions"].values())},indent=2))
    summary["total_samples"]=sum(v.get("n",0) for v in summary["regions"].values())
    return summary

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(retrain_and_save(),indent=2))
