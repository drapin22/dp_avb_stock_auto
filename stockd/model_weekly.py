from __future__ import annotations

from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model
from stockd.telegram_utils import send_chunked_message, send_telegram_document


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

    dfs = []
    for path, region_default in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)

        if "Active" in df.columns:
            df = df[df["Active"] == 1]

        if "Region" not in df.columns:
            df["Region"] = region_default

        if "Ticker" not in df.columns:
            raise ValueError(f"{path} must contain a 'Ticker' column.")

        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["Region"] = df["Region"].astype(str).str.upper().str.strip()
        dfs.append(df[["Ticker","Region"]])

    if not dfs:
        return pd.DataFrame(columns=["Ticker","Region"])

    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def load_prices_history() -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date","Ticker","Region","Currency","Close"])
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df = df.dropna(subset=["Date","Ticker","Region","Close"])
    return df.sort_values(["Ticker","Region","Date"])


def run_stockd_weekly_model() -> None:
    today = date.today()
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)
    horizon_days = 5

    holdings = load_all_holdings()
    if holdings.empty:
        send_chunked_message("StockD weekly forecast: no holdings.")
        return

    prices_history = load_prices_history()

    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    merged = holdings.merge(preds, on=["Ticker","Region"], how="left")
    merged["ER_Pct"] = pd.to_numeric(merged["ER_Pct"], errors="coerce").fillna(0.0)

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
            "Notes": settings.FORECAST_NOTES,
        })

    new_df = pd.DataFrame(rows)

    if settings.FORECASTS_FILE.exists():
        old = pd.read_csv(settings.FORECASTS_FILE)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["Date","Ticker","TargetDate","ModelVersion","Region"], keep="last", inplace=True)
        combined.to_csv(settings.FORECASTS_FILE, index=False)
    else:
        new_df.to_csv(settings.FORECASTS_FILE, index=False)

    # export weekly csv in reports for attaching
    weekly_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start}_{target_date}.csv"
    new_df.to_csv(weekly_csv, index=False)

    # telegram message: all tickers
    latest = {}
    if not prices_history.empty:
        last = prices_history.groupby(["Ticker","Region"], as_index=False).tail(1)
        for _, r in last.iterrows():
            latest[(r["Ticker"], r["Region"])] = float(r["Close"])

    lines = [
        "StockD weekly forecast",
        f"Week: {week_start} → {target_date}",
        f"Model: {settings.MODEL_VERSION_TAG}",
        "",
        "Ticker (Reg): Price → Target | ER%",
    ]

    for _, r in new_df.sort_values(["Region","Ticker"]).iterrows():
        t = r["Ticker"]; reg = r["Region"]
        er = float(r["ER_Pct"])
        p = latest.get((t, reg), None)
        if p is None:
            lines.append(f"- {t} ({reg}): no price | {er:+.2f}%")
        else:
            tgt = p * (1.0 + er/100.0)
            lines.append(f"- {t} ({reg}): {p:.2f} → {tgt:.2f} | {er:+.2f}%")

    send_chunked_message("\n".join(lines))
    send_telegram_document(weekly_csv, caption="Full weekly forecast (CSV)")


if __name__ == "__main__":
    run_stockd_weekly_model()
