# stockd/weekly_report.py
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests

from stockd import settings


FORECASTS_PATH = settings.DATA_DIR / "forecasts_stockd.csv"
PRICES_PATH = settings.PRICES_HISTORY

# Optional env vars pentru Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


@dataclass
class WeeklyReportInfo:
    week_start: date
    target_date: date
    csv_path: Path
    xlsx_path: Path
    plot_path: Path
    n_forecasts: int
    n_with_real: int


# ----------------- UTILITARE DE ÎNCĂRCARE -----------------


def _load_forecasts() -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"Forecast file not found: {FORECASTS_PATH}")

    df = pd.read_csv(FORECASTS_PATH)
    if df.empty:
        raise ValueError("Forecast file is empty.")

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["WeekStart"] = pd.to_datetime(df["WeekStart"]).dt.date
    df["TargetDate"] = pd.to_datetime(df["TargetDate"]).dt.date
    return df


def _load_prices() -> pd.DataFrame:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices file not found: {PRICES_PATH}")

    df = pd.read_csv(PRICES_PATH)
    if df.empty:
        raise ValueError("Prices file is empty.")

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


# ----------------- LOGICĂ: RANDAMENT REAL SĂPTĂMÂNĂ -----------------


def _compute_weekly_returns(
    prices: pd.DataFrame,
    week_start: date,
    target_date: date,
) -> pd.DataFrame:
    """
    Calculează randamentul REAL pe săptămână:
        Realized_Pct = (ultima_cotație / prima_cotație - 1) * 100
    pentru fiecare (Ticker, Region) în intervalul [week_start, target_date].
    """

    mask = (prices["Date"] >= week_start) & (prices["Date"] <= target_date)
    df = prices.loc[mask].copy()

    if df.empty:
        # întoarcem frame gol dar cu coloanele corecte
        return pd.DataFrame(columns=["Ticker", "Region", "Realized_Pct"])

    # Sortăm pentru a putea lua first/last corect
    df.sort_values(["Ticker", "Region", "Date"], inplace=True)

    first = (
        df.groupby(["Ticker", "Region"])
        .first()
        .reset_index()[["Ticker", "Region", "Close"]]
        .rename(columns={"Close": "StartPrice"})
    )

    last = (
        df.groupby(["Ticker", "Region"])
        .last()
        .reset_index()[["Ticker", "Region", "Close"]]
        .rename(columns={"Close": "EndPrice"})
    )

    merged = first.merge(last, on=["Ticker", "Region"], how="inner")
    merged["Realized_Pct"] = (merged["EndPrice"] / merged["StartPrice"] - 1.0) * 100.0

    return merged[["Ticker", "Region", "Realized_Pct"]]


# ----------------- GENERARE RAPORT + GRAFIC -----------------


def run_report(today: Optional[date] = None) -> WeeklyReportInfo:
    if today is None:
        today = date.today()

    forecasts = _load_forecasts()
    prices = _load_prices()

    # Alegem cea mai recentă săptămână pentru care avem forecast
    latest_week = forecasts["WeekStart"].max()
    week_rows = forecasts[forecasts["WeekStart"] == latest_week]

    target_date = week_rows["TargetDate"].max()
    f_week = week_rows[week_rows["TargetDate"] == target_date].copy()

    if f_week.empty:
        raise ValueError("No forecasts found for latest week.")

    # Calculează randamentele reale
    returns = _compute_weekly_returns(prices, latest_week, target_date)

    # !!! Aici era KeyError: acum returns are garantat Ticker + Region
    merged = f_week.merge(returns, on=["Ticker", "Region"], how="left")

    merged["Realized_Pct"] = merged["Realized_Pct"].fillna(0.0)
    merged["Error_Pct"] = merged["Realized_Pct"] - merged["ER_Pct"]

    # sortare pentru raport: cele mai bune / mai slabe predicții
    merged.sort_values("ER_Pct", ascending=False, inplace=True)

    # Director de output
    out_dir = settings.DATA_DIR / "weekly_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = latest_week.strftime("%Y%m%d")
    csv_path = out_dir / f"weekly_report_{stamp}.csv"
    xlsx_path = out_dir / f"weekly_report_{stamp}.xlsx"
    plot_path = out_dir / f"weekly_plot_{stamp}.png"

    # Salvăm CSV
    merged.to_csv(csv_path, index=False)

    # Salvăm Excel
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="Model_vs_Real")

    # Grafic simplu: ER_Pct vs Realized_Pct
    _plot_forecast_vs_real(merged, plot_path, latest_week, target_date)

    info = WeeklyReportInfo(
        week_start=latest_week,
        target_date=target_date,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        plot_path=plot_path,
        n_forecasts=len(f_week),
        n_with_real=(merged["Realized_Pct"] != 0.0).sum(),
    )

    print(
        f"[REPORT] Generated weekly report for {latest_week} -> {target_date} "
        f"(tickers={info.n_forecasts}, with_real={info.n_with_real})"
    )

    return info


def _plot_forecast_vs_real(
    df: pd.DataFrame,
    path: Path,
    week_start: date,
    target_date: date,
) -> None:
    # luăm top 10 ca să nu iasă graficul ilizibil
    top = df.copy().sort_values("ER_Pct", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    x = range(len(top))

    plt.bar(x, top["ER_Pct"], width=0.4, label="Model ER %")
    plt.bar(
        [i + 0.4 for i in x],
        top["Realized_Pct"],
        width=0.4,
        label="Realized %"
    )

    plt.xticks([i + 0.2 for i in x], top["Ticker"], rotation=45)
    plt.ylabel("Return %")
    plt.title(f"StockD weekly forecast vs real ({week_start} → {target_date})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[REPORT] Saved plot to {path}")


# ----------------- TELEGRAM -----------------


def _telegram_base_url() -> Optional[str]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID, skipping.")
        return None
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _send_telegram_text(message: str) -> None:
    base = _telegram_base_url()
    if base is None:
        return

    url = f"{base}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print("[TELEGRAM] Sent text message.")
    except Exception as exc:
        print(f"[TELEGRAM] ERROR sending text: {exc}")


def _send_telegram_document(path: Path, caption: str = "") -> None:
    base = _telegram_base_url()
    if base is None:
        return

    url = f"{base}/sendDocument"
    try:
        with path.open("rb") as f:
            files = {"document": (path.name, f)}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=30)
            r.raise_for_status()
        print(f"[TELEGRAM] Sent document: {path.name}")
    except Exception as exc:
        print(f"[TELEGRAM] ERROR sending document {path}: {exc}")


def _send_telegram_photo(path: Path, caption: str = "") -> None:
    base = _telegram_base_url()
    if base is None:
        return

    url = f"{base}/sendPhoto"
    try:
        with path.open("rb") as f:
            files = {"photo": (path.name, f)}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=30)
            r.raise_for_status()
        print(f"[TELEGRAM] Sent photo: {path.name}")
    except Exception as exc:
        print(f"[TELEGRAM] ERROR sending photo {path}: {exc}")


# ----------------- ORCHESTRATOR -----------------


def run_report_and_notify() -> None:
    info = run_report()

    msg = (
        f"*StockD – Weekly evaluation*\n"
        f"Week: `{info.week_start}` → `{info.target_date}`\n"
        f"Tickers in forecast: *{info.n_forecasts}*\n"
        f"With realized prices: *{info.n_with_real}*\n\n"
        f"Files:\n"
        f"- CSV: `{info.csv_path.name}`\n"
        f"- Excel: `{info.xlsx_path.name}`\n"
        f"- Plot: `{info.plot_path.name}`"
    )

    _send_telegram_text(msg)
    _send_telegram_document(info.csv_path, caption="Weekly report (CSV)")
    _send_telegram_document(info.xlsx_path, caption="Weekly report (Excel)")
    _send_telegram_photo(info.plot_path, caption="Forecast vs Realized returns")


if __name__ == "__main__":
    run_report_and_notify()
