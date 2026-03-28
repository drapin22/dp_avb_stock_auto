from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from stockd import settings

log = logging.getLogger(__name__)

FEATURES_RO = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","volume_ratio","peer_rel_20d","regime","vix_level","bench_ret_5d","rsi_14","z_price_60d","days_to_exdiv_norm","post_exdiv_flag","earnings_proximity"]
FEATURES_EU = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","volume_ratio","peer_rel_20d","regime","vix_level","dxy_ret_5d","bench_ret_5d","rsi_14","z_price_60d","earnings_proximity"]
FEATURES_US = ["ret_5d","ret_20d","ret_60d","vol_20d","max_dd_60d","beta","volume_ratio","peer_rel_20d","regime","vix_level","dxy_ret_5d","bench_ret_5d","rsi_14","z_price_60d","earnings_proximity"]
REGION_FEATURES = {"RO": FEATURES_RO, "EU": FEATURES_EU, "US": FEATURES_US}
MIN_TRAIN_WEEKS = 8
RIDGE_ALPHA = 0.5

def _rsi(close, period=14):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    gain = d.clip(lower=0).tail(period).mean()
    loss = (-d.clip(upper=0)).tail(period).mean()
    if loss == 0: return 100.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))

def _load_dividend_calendar():
    p = settings.DATA_DIR / "dividend_calendar.csv"
    if not p.exists(): return pd.DataFrame(columns=["Ticker","ExDivDate"])
    df = pd.read_csv(p); df["ExDivDate"] = pd.to_datetime(df["ExDivDate"], errors="coerce")
    return df.dropna(subset=["ExDivDate"])

def _load_earnings_calendar():
    p = settings.DATA_DIR / "earnings_calendar.csv"
    if not p.exists(): return pd.DataFrame(columns=["Ticker","EarningsDate"])
    df = pd.read_csv(p); df["EarningsDate"] = pd.to_datetime(df["EarningsDate"], errors="coerce")
    return df.dropna(subset=["EarningsDate"])

def compute_features_as_of(prices, as_of, div_cal, earn_cal, region_filter=None):
    p = prices[prices["Date"] <= as_of].copy()
    if region_filter: p = p[p["Region"] == region_filter]
    if p.empty: return pd.DataFrame()
    proxy = p.groupby(["Region","Date"])["Close"].mean().pct_change().rename("ProxyRet").reset_index()
    reg_ret20 = {}
    for reg, g in p.groupby("Region"):
        vals = [((float(tg.sort_values("Date")["Close"].reset_index(drop=True).iloc[-1]) / float(tg.sort_values("Date")["Close"].reset_index(drop=True).iloc[-21]) - 1.0)*100.0) for _, tg in g.groupby("Ticker") if len(tg) >= 21]
        reg_ret20[reg] = float(np.nanmean(vals)) if vals else 0.0
    has_vol = "Volume" in p.columns
    rows = []
    for (tkr, reg), g in p.groupby(["Ticker","Region"]):
        g = g.sort_values("Date"); close = g["Close"].astype(float).reset_index(drop=True); n = len(close)
        def sr(lb): return (float(close.iloc[-1])/float(close.iloc[-lb-1])-1.0)*100.0 if n>=lb+1 else 0.0
        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std()*np.sqrt(252)*100.0) if len(rets)>=10 else 20.0
        win = close.tail(60) if n>=10 else close
        max_dd = float(((win/win.cummax())-1.0).min()*100.0)
        prx = proxy[proxy["Region"]==reg].set_index("Date")["ProxyRet"]
        ar = rets.copy(); ar.index = g["Date"].iloc[1:len(rets)+1].values; ar = ar.reindex(prx.index).dropna()
        beta = 1.0
        if len(ar) >= 10:
            vb = float(prx.reindex(ar.index).var())
            if vb > 1e-12: beta = float(ar.cov(prx.reindex(ar.index)) / vb)
        vol_ratio = 1.0
        if has_vol:
            vc = pd.to_numeric(g["Volume"],errors="coerce").astype(float).reset_index(drop=True)
            avg20 = float(vc.tail(20).mean())
            if avg20 > 0: vol_ratio = float(vc.iloc[-1]/avg20)
        ret_20d = sr(20); peer_rel = ret_20d - reg_ret20.get(reg,0.0)
        sma60 = float(close.tail(60).mean()) if n>=20 else float(close.mean())
        regime = 1.0 if float(close.iloc[-1]) > sma60 else 0.0
        rsi_14 = _rsi(close, 14)
        z_price = (float(close.iloc[-1])-float(close.tail(60).mean()))/(float(close.tail(60).std())+1e-8) if n>=20 else 0.0
        days_to_exdiv_norm = post_exdiv_flag = 0.0
        if not div_cal.empty:
            td = div_cal[div_cal["Ticker"]==tkr].copy()
            if not td.empty:
                td["diff"] = (td["ExDivDate"]-as_of).dt.days
                fut = td[td["diff"]>=0]; past = td[td["diff"]<0]
                if not fut.empty: days_to_exdiv_norm = max(0.0, 1.0-int(fut["diff"].min())/30.0)
                if not past.empty and abs(int(past["diff"].max())) <= 5: post_exdiv_flag = 1.0
        earnings_proximity = 0.0
        if not earn_cal.empty:
            te = earn_cal[earn_cal["Ticker"]==tkr].copy()
            if not te.empty:
                te["diff"] = (te["EarningsDate"]-as_of).dt.days.abs()
                earnings_proximity = max(0.0, 1.0-int(te["diff"].min())/10.0)
        rows.append({"Ticker":tkr,"Region":reg,"ret_5d":sr(5),"ret_20d":ret_20d,"ret_60d":sr(60),"vol_20d":vol_20d,"max_dd_60d":max_dd,"beta":beta,"volume_ratio":vol_ratio,"peer_rel_20d":peer_rel,"regime":regime,"vix_level":0.0,"dxy_ret_5d":0.0,"bench_ret_5d":0.0,"rsi_14":rsi_14,"z_price_60d":z_price,"days_to_exdiv_norm":days_to_exdiv_norm,"post_exdiv_flag":post_exdiv_flag,"earnings_proximity":earnings_proximity})
    return pd.DataFrame(rows)

def compute_forward_return(prices, as_of, horizon=5):
    rows = []
    for (tkr,reg), g in prices.groupby(["Ticker","Region"]):
        g = g.sort_values("Date"); past = g[g["Date"]<=as_of]; future = g[g["Date"]>as_of].head(horizon)
        if past.empty or future.empty: continue
        p0,p1 = float(past.iloc[-1]["Close"]), float(future.iloc[-1]["Close"])
        if p0>0: rows.append({"Ticker":tkr,"Region":reg,"fwd_ret":(p1/p0-1.0)*100.0,"AsOf":as_of})
    return pd.DataFrame(rows)

def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    A = X.T @ X + alpha*np.eye(X.shape[1])
    try: coef = np.linalg.solve(A, X.T @ y)
    except: coef = np.zeros(X.shape[1])
    return coef, float(np.mean(y) - X.mean(axis=0) @ coef)

def xgb_available():
    try: import xgboost; return True
    except: return False

def xgb_fit(X_tr, y_tr, X_val=None, y_val=None):
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_alpha=0.5,reg_lambda=1.0,random_state=42,verbosity=0)
        if X_val is not None: m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
        else: m.fit(X_tr,y_tr)
        return m
    except: return None

def run_walkforward_region(prices, region, div_cal, earn_cal, horizon=5):
    feat_cols = REGION_FEATURES[region]
    p = prices[prices["Region"]==region].copy(); p["Date"] = pd.to_datetime(p["Date"])
    mondays = sorted(set(d for d in p["Date"].unique() if pd.Timestamp(d).dayofweek==0))
    train_feats=[]; eval_rows=[]; cur_coef=np.zeros(len(feat_cols)); cur_int=0.1; cur_type="ridge"; cur_xgb=None
    for i,monday in enumerate(mondays[:-1]):
        ts=pd.Timestamp(monday)
        feats=compute_features_as_of(prices,ts,div_cal,earn_cal,region)
        fwds=compute_forward_return(prices,ts,horizon)
        if feats.empty or fwds.empty: continue
        merged=feats.merge(fwds,on=["Ticker","Region"],how="left"); merged["AsOf"]=ts
        for c in feat_cols:
            if c not in merged.columns: merged[c]=0.0
        train_feats.append(merged)
        if i>=MIN_TRAIN_WEEKS and len(train_feats)>=2:
            tr=pd.concat(train_feats[:-1],ignore_index=True).dropna(subset=feat_cols+["fwd_ret"])
            val=train_feats[-1].dropna(subset=feat_cols+["fwd_ret"])
            if len(tr)>=4 and len(val)>=2:
                Xtr=tr[feat_cols].fillna(0.0).values; ytr=tr["fwd_ret"].values
                Xv=val[feat_cols].fillna(0.0).values; yv=val["fwd_ret"].values
                rc,ri=ridge_fit(Xtr,ytr); rp=ri+Xv@rc; rmae=float(np.mean(np.abs(rp-yv)))
                cur_coef,cur_int,cur_type,cur_xgb=rc,ri,"ridge",None
                if xgb_available() and len(Xtr)>=20:
                    xm=xgb_fit(Xtr,ytr,Xv,yv)
                    if xm is not None:
                        xp=xm.predict(Xv); xmae=float(np.mean(np.abs(xp-yv)))
                        if xmae < rmae*0.95: cur_type="xgboost"; cur_xgb=xm
            Xp=merged[feat_cols].fillna(0.0).values
            preds=(cur_xgb.predict(Xp) if cur_type=="xgboost" and cur_xgb else cur_int+Xp@cur_coef)
            for j,(_,row) in enumerate(merged.iterrows()):
                pred=float(preds[j]); actual=float(row["fwd_ret"]) if not pd.isna(row.get("fwd_ret")) else float("nan")
                err=(pred-actual) if not np.isnan(actual) else float("nan")
                eval_rows.append({"WeekStart":str(ts.date()),"Ticker":row["Ticker"],"Region":region,"Predicted_ER":round(pred,4),"Actual_ER":round(actual,4) if not np.isnan(actual) else None,"Error_Pct":round(err,4) if not np.isnan(err) else None,"AbsError_Pct":round(abs(err),4) if not np.isnan(err) else None,"DirectionHit":1.0 if pred*actual>0 else (0.0 if not np.isnan(actual) else None),"ModelType":cur_type})
    all_df=pd.concat(train_feats,ignore_index=True).dropna(subset=feat_cols+["fwd_ret"]) if train_feats else pd.DataFrame()
    fc,fi,ft,fx=cur_coef,cur_int,cur_type,cur_xgb
    if len(all_df)>=4:
        Xa=all_df[feat_cols].fillna(0.0).values; ya=all_df["fwd_ret"].values
        rc,ri=ridge_fit(Xa,ya); fc,fi,ft,fx=rc,ri,"ridge",None
        if xgb_available() and len(Xa)>=20:
            sp=int(len(Xa)*0.8)
            if sp>=4 and len(Xa)-sp>=2:
                xm=xgb_fit(Xa[:sp],ya[:sp],Xa[sp:],ya[sp:])
                if xm is not None:
                    rp=fi+Xa[sp:]@fc; xp=xm.predict(Xa[sp:])
                    if np.mean(np.abs(xp-ya[sp:]))<np.mean(np.abs(rp-ya[sp:]))*0.95:
                        xmf=xgb_fit(Xa,ya); ft,fx="xgboost",(xmf if xmf else xm)
    pseudo_coef = fx.feature_importances_.tolist() if ft=="xgboost" and fx is not None else fc.tolist()
    state={"region":region,"feature_cols":feat_cols,"coef":pseudo_coef,"intercept":float(fi),"model_type":ft,"trained_on_target_date":str(mondays[-2].date() if len(mondays)>=2 else mondays[-1].date() if mondays else ""),"n_samples":int(len(all_df))}
    if ft=="xgboost" and fx is not None:
        try:
            mp=settings.DATA_DIR/f"xgb_model_{region}.json"; fx.save_model(str(mp)); state["xgb_model_path"]=str(mp)
        except: pass
    return pd.DataFrame(eval_rows), state

def retrain_and_save(prices_path=None):
    from stockd.scoring import compute_scores, save_scores
    prices_path=prices_path or settings.PRICES_HISTORY
    if not prices_path.exists(): return {"error":f"Prices not found: {prices_path}"}
    prices=pd.read_csv(prices_path)
    prices["Date"]=pd.to_datetime(prices["Date"],errors="coerce"); prices["Close"]=pd.to_numeric(prices["Close"],errors="coerce")
    prices["Ticker"]=prices["Ticker"].astype(str).str.upper().str.strip(); prices["Region"]=prices["Region"].astype(str).str.upper().str.strip()
    prices=prices.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])
    div_cal=_load_dividend_calendar(); earn_cal=_load_earnings_calendar()
    all_eval=[]; all_states={}; summary={"regions":{}}
    for region in ["RO","EU","US"]:
        log.info(f"Training {region} model...")
        eval_df,state=run_walkforward_region(prices,region,div_cal,earn_cal)
        all_states[region]=state
        mp=settings.DATA_DIR/f"model_state_{region}.json"; mp.write_text(json.dumps(state,indent=2,ensure_ascii=False))
        log.info(f"  {region}: n={state['n_samples']}, type={state['model_type']}")
        if not eval_df.empty: all_eval.append(eval_df)
        summary["regions"][region]={"n_samples":state["n_samples"],"model_type":state["model_type"],"n_eval":len(eval_df)}
    combined={"feature_cols":FEATURES_RO,"coef":all_states.get("RO",{}).get("coef",[0.0]*len(FEATURES_RO)),"intercept":all_states.get("RO",{}).get("intercept",0.1),"model_type":"per_region","trained_on_target_date":str(pd.Timestamp.now().date()),"n_samples":sum(s.get("n_samples",0) for s in all_states.values()),"per_region":{r:s for r,s in all_states.items()}}
    (settings.DATA_DIR/"model_state.json").write_text(json.dumps(combined,indent=2,ensure_ascii=False))
    if all_eval:
        eval_df=pd.concat(all_eval,ignore_index=True); eval_path=settings.DATA_DIR/"backtest_eval.csv"
        eval_df.to_csv(eval_path,index=False); summary["n_eval_rows"]=len(eval_df)
        scores=compute_scores(eval_df)
        if not scores.empty: save_scores(scores); summary["n_scored"]=len(scores)
    ro=all_states.get("RO",{}); coef=ro.get("coef",[]); names=ro.get("feature_cols",[])
    summary["top_features_RO"]={k:round(v,4) for k,v in sorted(zip(names,coef),key=lambda x:abs(x[1]),reverse=True)[:6]}
    return summary

if __name__=="__main__":
    import sys; logging.basicConfig(level=logging.INFO); print(json.dumps(retrain_and_save(),indent=2))
