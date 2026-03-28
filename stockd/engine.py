from __future__ import annotations
from datetime import date
from typing import Dict, Tuple
import json
import numpy as np
import pandas as pd
from stockd import settings
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.calibration import load_calibration
from stockd.scoring import load_scores

def _safe_float(x, default=0.0):
    try: return default if pd.isna(x) else float(x)
    except: return default

def _load_region_state(region):
    path = settings.DATA_DIR / f"model_state_{region}.json"
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except: pass
    fb = settings.DATA_DIR / "model_state.json"
    if fb.exists():
        try:
            c = json.loads(fb.read_text(encoding="utf-8"))
            if "per_region" in c and region in c["per_region"]: return c["per_region"][region]
        except: pass
    return {"feature_cols":[],"coef":[],"intercept":0.1,"model_type":"ridge","n_samples":0}

_XGB_CACHE = {}

def _load_xgb(region):
    try:
        import xgboost as xgb
        mp = settings.DATA_DIR / f"xgb_model_{region}.json"
        if mp.exists(): m=xgb.XGBRegressor(); m.load_model(str(mp)); return m
    except: pass
    return None

def _predict_region(df, region):
    state = _load_region_state(region)
    feat_cols = state.get("feature_cols") or []
    if not feat_cols: return np.zeros(len(df))
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    X = df[feat_cols].values.astype(float)
    if state.get("model_type") == "xgboost":
        if region not in _XGB_CACHE: _XGB_CACHE[region] = _load_xgb(region)
        xm = _XGB_CACHE.get(region)
        if xm is not None:
            try: return xm.predict(X)
            except: pass
    coef = np.array(state.get("coef") or [0.0]*len(feat_cols), dtype=float)
    intercept = float(state.get("intercept") or 0.1)
    if len(coef) != X.shape[1]: return np.full(len(df), intercept)
    return intercept + X @ coef

def _regime_shrink(er, macro, stock_regime=1.0):
    vix=_safe_float(macro.get("vix_level"),0.0); dxy=_safe_float(macro.get("dxy_ret_5d"),0.0)
    shrink=1.0
    if vix>=28: shrink*=0.70
    elif vix>=22: shrink*=0.85
    if dxy>=1.5: shrink*=0.80
    elif dxy>=0.8: shrink*=0.90
    if stock_regime<0.5 and er>0: shrink*=0.75
    return er*shrink, {"shrink":round(shrink,3),"vix":vix,"dxy":dxy}

def run_stockd_model(holdings, prices_history, as_of=None, horizon_days=5):
    as_of = as_of or date.today(); as_of_ts = pd.Timestamp(as_of)
    if holdings is None or holdings.empty: return pd.DataFrame(columns=["Ticker","Region","ER_Pct"])
    h = holdings.copy()
    h["Ticker"]=h["Ticker"].astype(str).str.upper().str.strip()
    h["Region"]=h["Region"].astype(str).str.upper().str.strip()
    h = h.drop_duplicates(subset=["Ticker","Region"])
    feats = compute_ticker_features(prices_history, as_of=as_of_ts)
    model_in = h.merge(feats, on=["Ticker","Region"], how="left")
    macro = get_macro_snapshot(as_of=as_of)
    for src in ["vix_level","dxy_ret_5d","oil_ret_5d","gold_ret_5d","bench_ret_5d"]:
        model_in[src] = _safe_float(macro.get(src), 0.0)
    defaults = {"volume_ratio":1.0,"peer_rel_20d":0.0,"regime":1.0,"ret_5d":0.0,"rsi_14":50.0,"z_price_60d":0.0,"days_to_exdiv_norm":0.0,"post_exdiv_flag":0.0,"earnings_proximity":0.0}
    for col,dv in defaults.items():
        if col not in model_in.columns: model_in[col]=dv
        else: model_in[col]=pd.to_numeric(model_in[col],errors="coerce").fillna(dv)
    calib=load_calibration(); calib_regions=(calib or {}).get("regions",{})
    scores=load_scores(); scores_map={}
    if scores is not None and not scores.empty:
        for _,r in scores.iterrows():
            scores_map[(str(r["Ticker"]).upper(),str(r["Region"]).upper())]={"score":_safe_float(r.get("score_0_100"),50.0),"tier":str(r.get("confidence_tier") or "").strip()}
    preds_all = np.zeros(len(model_in))
    for region in ["RO","EU","US"]:
        mask = model_in["Region"]==region
        if mask.sum()==0: continue
        sub = model_in[mask].copy()
        preds_all[mask.values] = _predict_region(sub, region)
    out_rows=[]
    for i,r in model_in.iterrows():
        idx=model_in.index.get_loc(i); t,reg=r["Ticker"],r["Region"]
        raw=float(preds_all[idx])
        rcfg=calib_regions.get(reg,{}); mult=_safe_float(rcfg.get("mult"),1.0); bias=_safe_float(rcfg.get("bias"),0.0); clip=_safe_float(rcfg.get("clip_pct"),5.0)
        cal=float(np.clip(bias+mult*raw,-abs(clip),abs(clip)))
        s=scores_map.get((t,reg)); reliability=float(np.clip(s["score"]/100.0,0.25,0.90)) if s else 0.50
        final=cal*reliability; regime_val=_safe_float(r.get("regime"),1.0)
        final,reginfo=_regime_shrink(final,macro,stock_regime=regime_val)
        final=float(np.clip(final,-5.0,5.0))
        rs=_load_region_state(reg)
        out_rows.append({"Ticker":t,"Region":reg,"Raw_ER_Pct":round(raw,4),"Calib_ER_Pct":round(cal,4),"Reliability":round(reliability,3),"RegimeShrink":reginfo.get("shrink",1.0),"volume_ratio":round(_safe_float(r.get("volume_ratio"),1.0),3),"peer_rel_20d":round(_safe_float(r.get("peer_rel_20d"),0.0),3),"rsi_14":round(_safe_float(r.get("rsi_14"),50.0),1),"regime":int(regime_val),"days_to_exdiv_norm":round(_safe_float(r.get("days_to_exdiv_norm"),0.0),3),"post_exdiv_flag":int(_safe_float(r.get("post_exdiv_flag"),0.0)),"earnings_proximity":round(_safe_float(r.get("earnings_proximity"),0.0),3),"score_0_100":round(s["score"],1) if s else 50.0,"confidence_tier":s["tier"] if s else "MEDIUM","ModelType":rs.get("model_type","ridge"),"ER_Pct":round(final,4)})
    return pd.DataFrame(out_rows).sort_values(["Region","Ticker"]).reset_index(drop=True)
