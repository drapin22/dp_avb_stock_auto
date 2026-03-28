from __future__ import annotations
import json
from datetime import date
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from stockd import settings
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.mentor import propose_news_deltas
from stockd.calibration import load_calibration
from stockd.scoring import load_scores

_CACHE: Dict = {}

def _sf(x, d=0.0):
    try: return d if pd.isna(x) else float(x)
    except: return d

def _load_region_state(region):
    if region in _CACHE: return _CACHE[region]
    path = settings.DATA_DIR / f"model_state_{region}.json"
    if path.exists():
        state = json.loads(path.read_text())
    else:
        from stockd.online_model import load_state
        state = load_state()
    _CACHE[region] = state
    return state

def _predict_region(df_region, region):
    state = _load_region_state(region)
    cols = list(state.get("feature_cols") or [])
    coef = np.array(state.get("coef") or [0.0]*len(cols), dtype=float)
    intercept = float(state.get("intercept", 0.1))
    X = df_region.copy()
    for c in cols:
        if c not in X.columns: X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    mat = X[cols].to_numpy(dtype=float) if cols else np.zeros((len(X),1))
    if mat.shape[1] == coef.shape[0]:
        return intercept + mat @ coef
    return np.full(len(df_region), intercept)

def _regime_shrink(er, macro, stock_regime=1.0):
    vix = _sf(macro.get("vix_level"),0.0); dxy = _sf(macro.get("dxy_ret_5d"),0.0)
    shrink = 1.0
    if vix>=28: shrink*=0.70
    elif vix>=22: shrink*=0.85
    if dxy>=1.5: shrink*=0.80
    elif dxy>=0.8: shrink*=0.90
    if stock_regime<0.5 and er>0: shrink*=0.75
    return er*shrink, {"shrink":round(shrink,3),"vix":vix,"dxy":dxy}

def run_stockd_model(holdings, prices_history, as_of=None, horizon_days=5):
    _CACHE.clear()
    as_of = as_of or date.today()
    as_of_ts = pd.Timestamp(as_of)
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["Ticker","Region","ER_Pct"])
    h = holdings.copy()
    h["Ticker"] = h["Ticker"].astype(str).str.upper().str.strip()
    h["Region"] = h["Region"].astype(str).str.upper().str.strip()
    h = h.drop_duplicates(subset=["Ticker","Region"])
    feats = compute_ticker_features(prices_history, as_of=as_of_ts)
    model_in = h.merge(feats, on=["Ticker","Region"], how="left")
    macro = get_macro_snapshot(as_of=as_of)
    for src in ["vix_level","dxy_ret_5d","oil_ret_5d","gold_ret_5d","bench_ret_5d"]:
        model_in[src] = _sf(macro.get(src),0.0)
    if "ro_proxy_ret_5d" in model_in.columns:
        mask = (model_in["Region"]=="RO") & (pd.to_numeric(model_in["bench_ret_5d"],errors="coerce").fillna(0.0)==0.0)
        model_in.loc[mask,"bench_ret_5d"] = pd.to_numeric(model_in.loc[mask,"ro_proxy_ret_5d"],errors="coerce").fillna(0.0)
    defaults = {"volume_ratio":1.0,"peer_rel_20d":0.0,"regime":1.0,"ret_5d":0.0,
                "rsi_14":50.0,"z_price_60d":0.0,"days_to_exdiv_norm":0.0,"post_exdiv_flag":0.0}
    for col, default in defaults.items():
        if col not in model_in.columns: model_in[col] = default
        else: model_in[col] = pd.to_numeric(model_in[col],errors="coerce").fillna(default)
    base_preds = np.zeros(len(model_in))
    idx_list = list(model_in.index)
    for region in ["RO","EU","US"]:
        mask = model_in["Region"]==region
        if mask.any():
            region_df = model_in[mask].reset_index(drop=True)
            preds = _predict_region(region_df, region)
            positions = [idx_list.index(i) for i in model_in[mask].index]
            for pos, p in zip(positions, preds):
                base_preds[pos] = float(p)
    deltas = {}
    if settings.ENABLE_LLM_NEWS_ADJ and settings.OPENAI_API_KEY:
        try:
            deltas = propose_news_deltas(tickers=model_in[["Ticker","Region"]].to_dict(orient="records"),macro=macro,as_of=as_of)
        except: pass
    calib = load_calibration(); calib_regions = (calib or {}).get("regions", {})
    scores = load_scores(); scores_map = {}
    if scores is not None and not scores.empty:
        for _, r in scores.iterrows():
            scores_map[(str(r["Ticker"]).upper().strip(), str(r["Region"]).upper().strip())] = {
                "score": _sf(r.get("score_0_100"),50.0), "tier": str(r.get("confidence_tier") or "").strip()}
    out_rows = []
    for pos, (i, r) in enumerate(model_in.iterrows()):
        t, reg = r["Ticker"], r["Region"]
        raw = float(base_preds[pos])
        delta = _sf(deltas.get((t,reg),0.0),0.0)
        pre_cal = raw+delta
        rcfg = calib_regions.get(reg,{})
        cal = float(np.clip(_sf(rcfg.get("bias"),0.0)+_sf(rcfg.get("mult"),1.0)*pre_cal,
                            -abs(_sf(rcfg.get("clip_pct"),5.0)),abs(_sf(rcfg.get("clip_pct"),5.0))))
        s = scores_map.get((t,reg))
        reliability = float(np.clip(s["score"]/100.0,0.25,0.90)) if s else 0.50
        final = cal*reliability
        regime_val = _sf(r.get("regime"),1.0)
        final, reginfo = _regime_shrink(final, macro, stock_regime=regime_val)
        final = float(np.clip(final,-5.0,5.0))
        out_rows.append({"Ticker":t,"Region":reg,
            "Raw_ER_Pct":round(raw,4),"News_Delta_PP":round(delta,4),
            "PreCalib_ER_Pct":round(pre_cal,4),"Calib_ER_Pct":round(cal,4),
            "Reliability":round(reliability,3),"RegimeShrink":reginfo.get("shrink",1.0),
            "volume_ratio":round(_sf(r.get("volume_ratio"),1.0),3),
            "peer_rel_20d":round(_sf(r.get("peer_rel_20d"),0.0),3),
            "regime":int(regime_val),"rsi_14":round(_sf(r.get("rsi_14"),50.0),1),
            "z_price_60d":round(_sf(r.get("z_price_60d"),0.0),3),
            "days_to_exdiv_norm":round(_sf(r.get("days_to_exdiv_norm"),0.0),3),
            "score_0_100":round(s["score"],1) if s else 50.0,
            "confidence_tier":s["tier"] if s else "MEDIUM","ER_Pct":round(final,4)})
    return pd.DataFrame(out_rows).sort_values(["Region","Ticker"]).reset_index(drop=True)
