# stockd/model_weekly.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import os
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model
from stockd.calibration import load_calibration, apply_calibration
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
        if not Path(path).exists():
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

    # Dedup corect: aceeași săptămână/ticker/regiune/model -> păstrează ultima Date
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined["WeekStart"] = pd.to_datetime(combined["WeekStart"], errors="coerce")
    combined["TargetDate"] = pd.to_datetime(combined["TargetDate"], errors="coerce")

    key = ["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"]
    combined = combined.sort_values("Date").groupby(key, as_index=False).tail(1).reset_index(drop=True)

    # back to strings (ca să păstrăm format CSV consistent)
    combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")
    combined["WeekStart"] = combined["WeekStart"].dt.strftime("%Y-%m-%d")
    combined["TargetDate"] = combined["TargetDate"].dt.strftime("%Y-%m-%d")

    combined.to_csv(forecasts_path, index=False)
    print(f"[MODEL] Saved forecasts to {forecasts_path} (rows={len(combined)})")


def _chunk_lines(lines: list[str], max_chars: int = 3500) -> list[str]:
    """
    Telegram are limită (aprox 4096). Folosim 3500 ca buffer safe.
    """
    chunks = []
    current = []
    current_len = 0
    for ln in lines:
        if current_len + len(ln) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = [ln]
            current_len = len(ln) + 1
        else:
            current.append(ln)
            current_len += len(ln) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


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

    merged = holdings.merge(preds[["Ticker", "Region", "ER_Pct"]], on=["Ticker", "Region"], how="left")
    merged["ER_Pct"] = pd.to_numeric(merged["ER_Pct"], errors="coerce").fillna(0.0)

    calib = load_calibration()
    merged = apply_calibration(merged, calib)

    # Construim rândurile de scris în CSV
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
            "Adj_ER_Pct": float(r.get("Adj_ER_Pct", r["ER_Pct"])),
            "Notes": settings.FORECAST_NOTES,
        })

    new_df = pd.DataFrame(rows)
    save_forecasts_append(new_df)

    # Salvează și un CSV “run-only” pentru Telegram (lista completă)
    run_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start.strftime('%Y-%m-%d')}_{target_date.strftime('%Y-%m-%d')}.csv"
    new_df.sort_values(["Region", "Adj_ER_Pct"], ascending=[True, False]).to_csv(run_csv, index=False)

    # Telegram: Top 10 + atașament cu toate
    try:
        msg_lines = [
            "*StockD weekly forecast*",
            f"Week: *{week_start}* → *{target_date}*",
            f"Universe: *{len(new_df)}* tickers",
            "",
            "Top 10 signals (Adj_ER_Pct):",
        ]
        top = new_df.sort_values(["Adj_ER_Pct"], ascending=False).head(10)
        for _, x in top.iterrows():
            msg_lines.append(
                f"- {x['Ticker']} ({x['Region']}): *{float(x['Adj_ER_Pct']):+.2f}%* (raw {float(x['ER_Pct']):+.2f}%)"
            )

        send_telegram_message("\n".join(msg_lines))

        # Atașament cu toate
        send_telegram_document(str(run_csv), caption=f"Full forecast list: {week_start} → {target_date}")

        # Opțional: trimite și restul tickere-lor în mesaje chunked
        rest = new_df.sort_values(["Adj_ER_Pct"], ascending=False).iloc[10:]
        if not rest.empty:
            lines = ["", "Remaining tickers (Adj_ER_Pct):"]
            for _, x in rest.iterrows():
                lines.append(f"- {x['Ticker']} ({x['Region']}): {float(x['Adj_ER_Pct']):+.2f}% (raw {float(x['ER_Pct']):+.2f}%)")

            chunks = _chunk_lines(lines)
            for chunk in chunks[:5]:  # limităm ca să nu spamăm
                send_telegram_message(chunk)

    except Exception as exc:
        print(f"[MODEL] Telegram notify failed: {exc}")


if __name__ == "__main__":
    run_stockd_weekly_model()
