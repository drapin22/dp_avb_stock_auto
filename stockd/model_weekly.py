from __future__ import annotations
from datetime import date, timedelta
import logging
import pandas as pd
from stockd import settings
from stockd.engine import run_stockd_model
from stockd.telegram_utils import send_chunked_message, send_telegram_document

log = logging.getLogger(__name__)


def get_next_monday(d):
    days = (7 - d.weekday()) % 7
    return d + timedelta(days=days or 7)


def load_all_holdings():
    sources = [(settings.HOLDINGS_RO,"RO"),(settings.HOLDINGS_EU,"EU"),(settings.HOLDINGS_US,"US")]
    dfs = []
    for path, reg in sources:
        if not path.exists(): continue
        df = pd.read_csv(path)
        if "Active" in df.columns: df = df[df["Active"] == 1]
        if "Region" not in df.columns: df["Region"] = reg
        if "Ticker" not in df.columns: continue
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["Region"] = df["Region"].astype(str).str.upper().str.strip()
        dfs.append(df[["Ticker","Region"]])
    if not dfs: return pd.DataFrame(columns=["Ticker","Region"])
    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def load_prices():
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date","Ticker","Region","Currency","Close"])
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"]   = pd.to_datetime(df["Date"],  errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Close"]  = pd.to_numeric(df.get("Close"), errors="coerce")
    return df.dropna(subset=["Date","Ticker","Region","Close"]).sort_values(["Ticker","Region","Date"])


def evaluate_last_week(prices):
    """Compare previous predictions to actual outcomes. Builds feedback loop."""
    fcast_path = settings.FORECASTS_FILE
    eval_path  = settings.DATA_DIR / "backtest_eval.csv"
    if not fcast_path.exists() or prices.empty: return 0
    forecasts = pd.read_csv(fcast_path)
    forecasts["Date"]       = pd.to_datetime(forecasts["Date"],       errors="coerce")
    forecasts["TargetDate"] = pd.to_datetime(forecasts["TargetDate"], errors="coerce")
    forecasts["ER_Pct"]     = pd.to_numeric(forecasts["ER_Pct"],      errors="coerce")
    today = pd.Timestamp(date.today())
    evaluable = forecasts[forecasts["TargetDate"] < today].copy()
    if evaluable.empty: return 0
    already = set()
    if eval_path.exists():
        ex = pd.read_csv(eval_path)
        for _, r in ex.iterrows():
            already.add((str(r.get("WeekStart","")), str(r.get("Ticker","")), str(r.get("Region",""))))
    new_rows = []
    for _, row in evaluable.iterrows():
        key = (str(row["Date"].date()), str(row["Ticker"]), str(row["Region"]))
        if key in already: continue
        t, reg = row["Ticker"], row["Region"]
        pred_er = float(row["ER_Pct"])
        tkr_p = prices[(prices["Ticker"]==t) & (prices["Region"]==reg)].sort_values("Date")
        p0r = tkr_p[tkr_p["Date"] <= row["Date"]].tail(1)
        p1r = tkr_p[tkr_p["Date"] <= row["TargetDate"]].tail(1)
        if p0r.empty or p1r.empty: continue
        p0, p1 = float(p0r.iloc[0]["Close"]), float(p1r.iloc[0]["Close"])
        if p0 <= 0: continue
        actual = (p1/p0-1.0)*100.0
        err = pred_er - actual
        new_rows.append({
            "WeekStart": str(row["Date"].date()), "Ticker": t, "Region": reg,
            "Predicted_ER": round(pred_er,4), "Actual_ER": round(actual,4),
            "Error_Pct": round(err,4), "AbsError_Pct": round(abs(err),4),
            "DirectionHit": 1.0 if pred_er*actual>0 else 0.0,
        })
    if not new_rows: return 0
    new_df = pd.DataFrame(new_rows)
    if eval_path.exists():
        old = pd.read_csv(eval_path)
        comb = pd.concat([old, new_df], ignore_index=True)
        comb.drop_duplicates(subset=["WeekStart","Ticker","Region"], keep="last", inplace=True)
        comb.to_csv(eval_path, index=False)
    else:
        new_df.to_csv(eval_path, index=False)
    log.info(f"Evaluated {len(new_rows)} predictions")
    return len(new_rows)


def update_scores():
    eval_path = settings.DATA_DIR / "backtest_eval.csv"
    if not eval_path.exists(): return 0
    try:
        from stockd.scoring import compute_scores, save_scores
        scores = compute_scores(pd.read_csv(eval_path))
        if not scores.empty:
            save_scores(scores)
            return len(scores)
    except Exception as e:
        log.warning(f"Score update failed: {e}")
    return 0


def maybe_retrain():
    try:
        from stockd.train import retrain_and_save
        result = retrain_and_save()
        log.info(f"Retrain: {result}")
        return result
    except Exception as e:
        log.warning(f"Retrain failed (non-fatal): {e}")
        return {}


def run_stockd_weekly_model():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today      = date.today()
    week_start = get_next_monday(today)
    target_dt  = week_start + timedelta(days=4)
    horizon    = 5

    holdings = load_all_holdings()
    if holdings.empty:
        send_chunked_message("StockD: no holdings found.")
        return

    prices = load_prices()

    # === SELF-IMPROVEMENT PIPELINE ===
    log.info("Step 1: Evaluating last week predictions")
    n_eval = evaluate_last_week(prices)

    log.info("Step 2: Updating confidence scores")
    n_scored = update_scores()

    log.info("Step 3: Re-training model")
    train_res = maybe_retrain()
    n_samples = train_res.get("n_samples", 0)
    top_feats = train_res.get("top_features", {})

    # === GENERATE SIGNALS ===
    log.info("Step 4: Generating weekly signals")
    preds  = run_stockd_model(holdings=holdings, prices_history=prices, as_of=today, horizon_days=horizon)
    merged = holdings.merge(preds, on=["Ticker","Region"], how="left")
    merged["ER_Pct"] = pd.to_numeric(merged.get("ER_Pct"), errors="coerce").fillna(0.0)

    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "Date": today.strftime("%Y-%m-%d"),
            "WeekStart": week_start.strftime("%Y-%m-%d"),
            "TargetDate": target_dt.strftime("%Y-%m-%d"),
            "ModelVersion": settings.MODEL_VERSION_TAG,
            "Ticker": r["Ticker"], "Region": r["Region"],
            "HorizonDays": horizon,
            "ER_Pct": float(r["ER_Pct"]),
            "Notes": settings.FORECAST_NOTES,
            "Adj_ER_Pct": float(r.get("Calib_ER_Pct", r["ER_Pct"])),
            "EngineStatus": "OK", "CalibApplied": "yes",
            "CalibReason": "ridge+calib+reliability+regime",
            "LastError": "",
            "RankSignal": "",
            "score_0_100": float(r.get("score_0_100", 50)),
            "confidence_tier": str(r.get("confidence_tier","MEDIUM")),
            "AdjERPct": float(r.get("Calib_ER_Pct", r["ER_Pct"])),
            "Reliability": float(r.get("Reliability", 0.5)),
            "VolScale": float(r.get("volume_ratio", 1.0)),
        })

    new_df = pd.DataFrame(rows)
    if settings.FORECASTS_FILE.exists():
        old = pd.read_csv(settings.FORECASTS_FILE)
        comb = pd.concat([old, new_df], ignore_index=True)
        comb.drop_duplicates(subset=["Date","Ticker","TargetDate","ModelVersion","Region"], keep="last", inplace=True)
        comb.to_csv(settings.FORECASTS_FILE, index=False)
    else:
        new_df.to_csv(settings.FORECASTS_FILE, index=False)

    weekly_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start}_{target_dt}.csv"
    new_df.to_csv(weekly_csv, index=False)

    # === TELEGRAM ===
    latest_p = {}
    if not prices.empty:
        for _, r in prices.groupby(["Ticker","Region"]).tail(1).iterrows():
            latest_p[(r["Ticker"], r["Region"])] = float(r["Close"])

    buys  = new_df[new_df["ER_Pct"] >  0.3]
    sells = new_df[new_df["ER_Pct"] < -0.3]
    holds = new_df[(new_df["ER_Pct"] >= -0.3) & (new_df["ER_Pct"] <= 0.3)]

    top_feat_str = " | ".join(f"{k}:{v:+.3f}" for k,v in list(top_feats.items())[:3]) if top_feats else "n/a"
    lines = [
        "StockD APEX V10.7F+ — Self-Improving Forecast",
        f"Week: {week_start} to {target_dt}",
        f"Training samples: {n_samples} | Evaluated: {n_eval} | Scored: {n_scored}",
        f"Top features: {top_feat_str}",
        f"Signals: {len(buys)} BUY | {len(sells)} SELL | {len(holds)} HOLD",
        "",
    ]
    for _, r in new_df.sort_values("ER_Pct", ascending=False).iterrows():
        t, reg = r["Ticker"], r["Region"]
        er = float(r["ER_Pct"])
        sc = float(r.get("score_0_100",50))
        sig = "BUY " if er > 0.3 else "SELL" if er < -0.3 else "HOLD"
        p = latest_p.get((t,reg))
        ps = f"{p:.2f}>{p*(1+er/100):.2f}" if p else "n/a"
        lines.append(f"  {sig} {t:8s}({reg}): {er:+.2f}%  [{ps}]  score={sc:.0f}")

    send_chunked_message("\n".join(lines))
    send_telegram_document(weekly_csv, caption="Full weekly forecast (CSV)")


if __name__ == "__main__":
    run_stockd_weekly_model()
