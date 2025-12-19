from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from stockd import settings
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.online_model import RidgeConfig, fit_ridge, save_state, load_state
from stockd.mentor import postmortem_and_rules
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def _load_forecasts() -> pd.DataFrame:
    if not settings.FORECASTS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.FORECASTS_FILE)
    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["ER_Pct"] = pd.to_numeric(df.get("ER_Pct"), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["WeekStart", "TargetDate", "Ticker", "Region"])
    return df


def _load_prices_history() -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Region", "Close"])
    return df.sort_values(["Ticker","Region","Date"])


def _asof_close(df: pd.DataFrame, asof_dt: pd.Timestamp) -> pd.DataFrame:
    d = df.copy()
    d = d[d["Date"] <= asof_dt].sort_values(["Ticker","Region","Date"])
    last = d.groupby(["Ticker","Region"], as_index=False).tail(1)
    return last[["Ticker","Region","Close"]].rename(columns={"Close": "Close_asof"})


def _realized_return(prices: pd.DataFrame, week_start: pd.Timestamp, week_end: pd.Timestamp) -> pd.DataFrame:
    p = prices[(prices["Date"] >= week_start) & (prices["Date"] <= week_end)].copy()
    if p.empty:
        return pd.DataFrame(columns=["Ticker","Region","RealizedReturnPct"])

    p = p.sort_values(["Ticker","Region","Date"])
    first = p.groupby(["Ticker","Region"])["Close"].first()
    last = p.groupby(["Ticker","Region"])["Close"].last()
    rr = (last / first - 1.0) * 100.0
    out = rr.rename("RealizedReturnPct").reset_index()
    return out


def run_learning() -> None:
    fc = _load_forecasts()
    px = _load_prices_history()

    if fc.empty or px.empty:
        send_telegram_message("StockD learning: missing forecasts or prices_history.")
        return

    # ultima săptămână închisă: targetdate max care e < azi
    today = pd.Timestamp(date.today())
    fc2 = fc[fc["TargetDate"] < today].copy()
    if fc2.empty:
        send_telegram_message("StockD learning: no completed weeks found in forecasts.")
        return

    last_target = fc2["TargetDate"].max()
    f_week = fc2[fc2["TargetDate"] == last_target].copy()
    week_start = pd.Timestamp(f_week["WeekStart"].iloc[0]).normalize()
    week_end = pd.Timestamp(last_target).normalize()

    # realized
    rr = _realized_return(px, week_start, week_end)
    if rr.empty:
        send_telegram_message("StockD learning: no realized prices for last completed week.")
        return

    # features as of week_start (folosim close <= week_start)
    holdings = f_week[["Ticker","Region"]].drop_duplicates()
    feat_df, _ = compute_ticker_features(px, holdings)

    # macro snapshot la startul săptămânii
    macro = get_macro_snapshot(datetime.combine(week_start.date(), datetime.min.time()))

    # map macro series în coloane
    def get_series(name, key):
        v = macro.get("series", {}).get(name, {}).get(key, None)
        return float(v) if v is not None else 0.0

    bench_ret_us = macro.get("regions", {}).get("US", {}).get("bench_ret_5d_pct", 0.0) or 0.0
    bench_ret_eu = macro.get("regions", {}).get("EU", {}).get("bench_ret_5d_pct", 0.0) or 0.0

    # join training
    train = f_week.merge(rr, on=["Ticker","Region"], how="left")
    train = train.merge(feat_df, on=["Ticker","Region"], how="left")

    train["vix_level"] = get_series("VIX", "last")
    train["dxy_ret_5d"] = get_series("DXY", "ret_5d_pct")
    train["oil_ret_5d"] = get_series("OIL", "ret_5d_pct")
    train["gold_ret_5d"] = get_series("GOLD", "ret_5d_pct")

    # region benchmark ret
    train["bench_ret_5d"] = 0.0
    train.loc[train["Region"] == "US", "bench_ret_5d"] = float(bench_ret_us)
    train.loc[train["Region"] == "EU", "bench_ret_5d"] = float(bench_ret_eu)
    # RO: folosim ro_proxy_ret_5d deja din features
    train.loc[train["Region"] == "RO", "bench_ret_5d"] = pd.to_numeric(train["ro_proxy_ret_5d"], errors="coerce").fillna(0.0)

    # umple NaN
    for c in ["ret_20d","ret_60d","vol_20d","max_dd_60d","beta","ro_proxy_ret_5d"]:
        train[c] = pd.to_numeric(train.get(c), errors="coerce").fillna(0.0)

    train["y"] = pd.to_numeric(train["RealizedReturnPct"], errors="coerce").fillna(0.0)

    # Fit ridge global (simplu, stabil)
    cfg = RidgeConfig(l2=10.0)
    state = fit_ridge(train, y_col="y", cfg=cfg)
    state["trained_on_week_end"] = str(week_end.date())
    state["trained_rows"] = int(len(train))

    save_state(settings.MODEL_STATE_JSON, state)

    # Mentor postmortem (safe rules)
    eval_rows = []
    for _, r in train.iterrows():
        eval_rows.append({
            "Ticker": r["Ticker"],
            "Region": r["Region"],
            "PredictedER": float(r["ER_Pct"]),
            "RealizedReturnPct": float(r["RealizedReturnPct"]),
            "ErrorPP": float(r["ER_Pct"] - r["RealizedReturnPct"]),
        })

    mentor = postmortem_and_rules(eval_rows, macro_snapshot=macro) if settings.ENABLE_MENTOR else {"ok": False, "reason": "MENTOR_DISABLED"}
    settings.MENTOR_OVERRIDES_JSON.write_text(json.dumps(mentor, indent=2, ensure_ascii=False), encoding="utf-8")

    # notify
    msg = [
        "StockD learning update",
        f"Week: {week_start.date()} → {week_end.date()}",
        f"Train rows: {len(train)}",
        f"Model state ok: {bool(state.get('ok'))}",
        f"Mentor ok: {bool(mentor.get('ok'))}",
    ]
    send_telegram_message("\n".join(msg))
    send_telegram_document(settings.MODEL_STATE_JSON, caption="model_state.json")
    send_telegram_document(settings.MENTOR_OVERRIDES_JSON, caption="mentor_overrides.json")


if __name__ == "__main__":
    run_learning()
