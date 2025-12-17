# stockd/model_weekly.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import os
import numpy as np
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model
from stockd.calibration import load_calibration, apply_calibration
from stockd.scoring import load_scores
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def get_next_monday(d: date) -> date:
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


def load_all_holdings() -> pd.DataFrame:
    sources = [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]
    dfs: list[pd.DataFrame] = []
    for path, region_default in sources:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Active" in df.columns:
            df = df[df["Active"] == 1]
        if "Region" not in df.columns:
            df["Region"] = region_default
        if "Ticker" not in df.columns:
            raise ValueError(f"{path} must contain a 'Ticker' column.")
        dfs.append(df[["Ticker", "Region"]])

    if not dfs:
        return pd.DataFrame(columns=["Ticker", "Region"])

    out = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    out["Ticker"] = out["Ticker"].astype(str)
    out["Region"] = out["Region"].astype(str)
    return out


def load_prices_history() -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def save_forecasts_append(new_df: pd.DataFrame) -> None:
    forecasts_path = settings.FORECASTS_FILE
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if forecasts_path.exists():
        old = pd.read_csv(forecasts_path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined["WeekStart"] = pd.to_datetime(combined["WeekStart"], errors="coerce")
    combined["TargetDate"] = pd.to_datetime(combined["TargetDate"], errors="coerce")

    key = ["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"]
    combined = combined.sort_values("Date").groupby(key, as_index=False).tail(1).reset_index(drop=True)

    combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")
    combined["WeekStart"] = combined["WeekStart"].dt.strftime("%Y-%m-%d")
    combined["TargetDate"] = combined["TargetDate"].dt.strftime("%Y-%m-%d")

    combined.to_csv(forecasts_path, index=False)


def run_stockd_weekly_model() -> None:
    today = date.today()
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)
    horizon_days = (target_date - week_start).days + 1

    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    prices_history = load_prices_history()

    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    merged = holdings.merge(
        preds[["Ticker", "Region", "ER_Pct", "EngineStatus", "LastError"]],
        on=["Ticker", "Region"],
        how="left",
    )
    merged["ER_Pct"] = pd.to_numeric(merged["ER_Pct"], errors="coerce").fillna(0.0)
    merged["EngineStatus"] = merged["EngineStatus"].fillna("UNKNOWN")
    merged["LastError"] = merged["LastError"].fillna("")

    engine_ok = bool((merged["EngineStatus"] == "OK").all())
    er_std = float(np.std(merged["ER_Pct"].values)) if len(merged) else 0.0
    low_signal = (er_std < 0.05)

    calib = load_calibration()
    if engine_ok and not low_signal:
        merged = apply_calibration(merged, calib)
    else:
        merged["Adj_ER_Pct"] = merged["ER_Pct"]
        merged["CalibApplied"] = False

    # join scores
    scores = load_scores()
    if not scores.empty:
        merged = merged.merge(scores[["Ticker", "Region", "score_0_100", "confidence_tier"]], on=["Ticker", "Region"], how="left")
    else:
        merged["score_0_100"] = np.nan
        merged["confidence_tier"] = np.nan

    merged["score_0_100"] = pd.to_numeric(merged["score_0_100"], errors="coerce").fillna(50.0)
    merged["confidence_tier"] = merged["confidence_tier"].fillna("MEDIUM")

    # “rank_signal” = Adj * (score/100) (conservator)
    merged["RankSignal"] = merged["Adj_ER_Pct"] * (merged["score_0_100"] / 100.0)

    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "Date": today.strftime("%Y-%m-%d"),
            "WeekStart": week_start.strftime("%Y-%m-%d"),
            "TargetDate": target_date.strftime("%Y-%m-%d"),
            "ModelVersion": settings.MODEL_VERSION_TAG,
            "Ticker": r["Ticker"],
            "Region": r["Region"],
            "HorizonDays": horizon_days,
            "ER_Pct": float(r["ER_Pct"]),
            "Adj_ER_Pct": float(r["Adj_ER_Pct"]),
            "EngineStatus": str(r.get("EngineStatus", "")),
            "LastError": str(r.get("LastError", "")),
            "score_0_100": float(r["score_0_100"]),
            "confidence_tier": str(r["confidence_tier"]),
            "RankSignal": float(r["RankSignal"]),
            "Notes": settings.FORECAST_NOTES,
        })

    new_df = pd.DataFrame(rows)
    save_forecasts_append(new_df)

    run_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start.strftime('%Y-%m-%d')}_{target_date.strftime('%Y-%m-%d')}.csv"
    new_df.sort_values(["Region", "RankSignal"], ascending=[True, False]).to_csv(run_csv, index=False)

    # Telegram
    header = [
        "*StockD weekly forecast*",
        f"Week: *{week_start}* → *{target_date}*",
        f"Universe: *{len(new_df)}* tickers",
    ]

    warnings = []
    if not engine_ok:
        warnings.append("Engine status: *FALLBACK* (OpenAI call failed or returned empty). Forecasts may be neutral.")
        le = str(merged["LastError"].iloc[0]) if len(merged) else ""
        if le:
            warnings.append(f"LastError: `{le[:180]}`")
    if low_signal and engine_ok:
        warnings.append("Model signal: *LOW VARIANCE* (near-constant outputs). Treat as low confidence.")
    if warnings:
        header += ["", "*Warnings:*"] + [f"- {w}" for w in warnings]

    lines = []
    lines += ["", "Top 10 signals (RankSignal):"]
    top = new_df.sort_values(["RankSignal"], ascending=False).head(10)
    for _, x in top.iterrows():
        lines.append(
            f"- {x['Ticker']} ({x['Region']}): *{float(x['Adj_ER_Pct']):+.2f}%* "
            f"(score {float(x['score_0_100']):.0f}, tier {x['confidence_tier']}, raw {float(x['ER_Pct']):+.2f}%)"
        )

    send_telegram_message("\n".join(header + lines))
    send_telegram_document(str(run_csv), caption=f"Full forecast list: {week_start} → {target_date}")


if __name__ == "__main__":
    run_stockd_weekly_model()
