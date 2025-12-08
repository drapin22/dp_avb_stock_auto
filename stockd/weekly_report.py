# stockd/weekly_report.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests

from stockd import settings


# ------------ Utilitare de dată / fișiere -----------------


def get_week_bounds(run_date: date) -> tuple[date, date]:
    """
    Pentru orice zi, întoarce (week_start, week_end) pentru săptămâna Luni–Vineri
    care S-A ÎNCHEIAT cel mai recent.

    - Dacă rulezi sâmbătă -> săptămâna de Luni–Vineri care tocmai s-a terminat
    - Dacă rulezi vineri -> săptămâna curentă (până astăzi)
    """
    # Câte zile au trecut de la ultimul vineri
    offset = (run_date.weekday() - 4) % 7
    week_end = run_date - timedelta(days=offset)     # vineri
    week_start = week_end - timedelta(days=4)        # luni
    return week_start, week_end


def get_reports_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# ------------ Încărcare date -----------------


def load_holdings_ro() -> pd.DataFrame:
    path = settings.HOLDINGS_RO
    if not path.exists():
        return pd.DataFrame(columns=["Ticker"])

    df = pd.read_csv(path)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]
    return df[["Ticker"]].drop_duplicates()


def load_prices_week_ro(week_start: date, week_end: date, tickers: list[str]) -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])

    mask = (
        (df["Region"] == "RO")
        & (df["Ticker"].isin(tickers))
        & (df["Date"] >= pd.Timestamp(week_start))
        & (df["Date"] <= pd.Timestamp(week_end))
    )
    df = df[mask].copy()
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def load_forecasts_ro(week_start: date, week_end: date) -> pd.DataFrame:
    """Forecasturile StockD pentru săptămâna respectivă (RO)."""
    forecasts_path = settings.DATA_DIR / "forecasts_stockd.csv"
    if not forecasts_path.exists():
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    f = pd.read_csv(forecasts_path)
    # Asigurăm formatele de dată
    for col in ["Date", "WeekStart", "TargetDate"]:
        if col in f.columns:
            f[col] = pd.to_datetime(f[col]).dt.date

    mask = (
        (f["Region"] == "RO")
        & (f["WeekStart"] == week_start)
        & (f["TargetDate"] == week_end)
    )
    f = f[mask].copy()

    if f.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    # Dacă există mai multe runde pentru același ticker în săptămâna asta,
    # păstrăm ultima (după Date).
    f.sort_values(["Ticker", "Date"], inplace=True)
    f = f.drop_duplicates(subset=["Ticker", "Region"], keep="last")

    return f[["Ticker", "Region", "ER_Pct"]]


# ------------ Calcul raport -----------------


def build_weekly_table(
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    week_start: date,
    week_end: date,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Region",
                "Mon_Close",
                "Tue_Close",
                "Wed_Close",
                "Thu_Close",
                "Fri_Close",
                "Actual_Weekly_Pct",
                "Model_ER_Pct",
                "Verdict",
            ]
        )

    # Pivot cu prețurile de închidere pe zile
    week_days = pd.date_range(week_start, week_end, freq="B")  # business days Mon–Fri
    pivot = prices.pivot_table(
        index=["Ticker", "Region"],
        columns="Date",
        values="Close",
        aggfunc="last",
    )

    # Asigurăm coloane pentru fiecare zi (chiar dacă lipsesc date – NaN)
    for d in week_days:
        if d not in pivot.columns:
            pivot[d] = np.nan

    pivot = pivot[sorted(pivot.columns)]  # ordonăm pe date

    # Redenumim în Mon/Tue/...
    rename_map = {}
    for d in week_days:
        col_name = d.strftime("%a")  # Mon, Tue, ...
        rename_map[d] = f"{col_name}_Close"
    pivot.rename(columns=rename_map, inplace=True)

    pivot.reset_index(inplace=True)

    # Calcul actual weekly %: ultima cotație vs prima din săptămână
    prices_sorted = prices.sort_values(["Ticker", "Date"])
    first = prices_sorted.groupby("Ticker")["Close"].first()
    last = prices_sorted.groupby("Ticker")["Close"].last()
    actual_weekly = (last / first - 1.0) * 100.0
    actual_df = actual_weekly.rename("Actual_Weekly_Pct").reset_index()

    # Unim pivotul cu actual_weekly
    merged = pivot.merge(actual_df, on="Ticker", how="left")

    # Unim cu forecasturile
    merged = merged.merge(forecasts, on=["Ticker", "Region"], how="left")
    merged.rename(columns={"ER_Pct": "Model_ER_Pct"}, inplace=True)

    # Verdict: HIT dacă semnul este același și niciunul nu e 0
    def verdict_row(row):
        a = row.get("Actual_Weekly_Pct", np.nan)
        m = row.get("Model_ER_Pct", np.nan)
        if pd.isna(a) or pd.isna(m):
            return "NO_DATA"
        if a == 0 or m == 0:
            return "NEUTRAL"
        if np.sign(a) == np.sign(m):
            return "HIT"
        return "MISS"

    merged["Verdict"] = merged.apply(verdict_row, axis=1)

    # Ordine coloane
    ordered_cols = [
        "Ticker",
        "Region",
        "Mon_Close",
        "Tue_Close",
        "Wed_Close",
        "Thu_Close",
        "Fri_Close",
        "Actual_Weekly_Pct",
        "Model_ER_Pct",
        "Verdict",
    ]
    # Adăugăm doar coloanele care chiar există
    ordered_cols = [c for c in ordered_cols if c in merged.columns]
    merged = merged[ordered_cols]

    return merged.sort_values("Ticker")


def portfolio_metrics(weekly_table: pd.DataFrame) -> dict:
    if weekly_table.empty:
        return {
            "n_names": 0,
            "hit_rate": np.nan,
            "mean_model_er": np.nan,
            "mean_actual": np.nan,
            "er_bias": np.nan,
        }

    valid = weekly_table[weekly_table["Verdict"].isin(["HIT", "MISS"])]
    n = len(valid)
    hits = (valid["Verdict"] == "HIT").sum()

    mean_model = valid["Model_ER_Pct"].mean()
    mean_actual = valid["Actual_Weekly_Pct"].mean()
    er_bias = mean_actual - mean_model

    return {
        "n_names": int(n),
        "hit_rate": float(hits / n) if n > 0 else np.nan,
        "mean_model_er": float(mean_model),
        "mean_actual": float(mean_actual),
        "er_bias": float(er_bias),
    }


# ------------ Telegram -----------------


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[TELEGRAM] Error {resp.status_code}: {resp.text}")
        else:
            print("[TELEGRAM] Message sent.")
    except Exception as exc:
        print(f"[TELEGRAM] Exception: {exc}")


# ------------ Main orchestration -----------------


def run_report(run_date: date | None = None) -> dict:
    if run_date is None:
        run_date = date.today()

    week_start, week_end = get_week_bounds(run_date)
    print(f"[REPORT] Week = {week_start} -> {week_end}")

    holdings_ro = load_holdings_ro()
    tickers = sorted(holdings_ro["Ticker"].unique().tolist())
    print(f"[REPORT] RO holdings count: {len(tickers)}")

    prices = load_prices_week_ro(week_start, week_end, tickers)
    print(f"[REPORT] Loaded {len(prices)} RO price rows for week.")

    forecasts = load_forecasts_ro(week_start, week_end)
    print(f"[REPORT] Loaded {len(forecasts)} RO forecasts for week.")

    weekly_table = build_weekly_table(prices, forecasts, week_start, week_end)
    metrics = portfolio_metrics(weekly_table)

    # Salvăm în Excel
    reports_dir = get_reports_dir()
    out_name = f"weekly_bvb_model_vs_real_{week_end.isoformat()}.xlsx"
    out_path = reports_dir / out_name

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        weekly_table.to_excel(writer, index=False, sheet_name="Per_Ticker")
        # sheet mic cu sumarul de portofoliu
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "WeekStart",
                    "WeekEnd",
                    "Names",
                    "HitRate",
                    "Mean_Model_ER_Pct",
                    "Mean_Actual_Pct",
                    "ER_Bias_Pct",
                ],
                "Value": [
                    week_start.isoformat(),
                    week_end.isoformat(),
                    metrics["n_names"],
                    metrics["hit_rate"],
                    metrics["mean_model_er"],
                    metrics["mean_actual"],
                    metrics["er_bias"],
                ],
            }
        )
        summary_df.to_excel(writer, index=False, sheet_name="Portfolio_Summary")

    print(f"[REPORT] Saved Excel report to {out_path}")

    result = {
        "week_start": week_start,
        "week_end": week_end,
        "weekly_table": weekly_table,
        "metrics": metrics,
        "excel_path": out_path,
    }
    return result


def format_telegram_summary(info: dict) -> str:
    ws = info["week_start"]
    we = info["week_end"]
    m = info["metrics"]

    hit_rate_pct = m["hit_rate"] * 100 if m["hit_rate"] == m["hit_rate"] else None  # NaN check
    text_lines = [
        f"*StockD – Weekly BVB Model vs Market*",
        f"Week: *{ws.isoformat()}* → *{we.isoformat()}*",
        "",
        f"Names evaluated: *{m['n_names']}*",
    ]

    if hit_rate_pct is not None:
        text_lines.append(f"Hit rate (sign): *{hit_rate_pct:.1f}%*")
    else:
        text_lines.append("Hit rate (sign): n/a")

    if m["mean_model_er"] == m["mean_model_er"]:
        text_lines.append(f"Mean model ER: *{m['mean_model_er']:.2f}%*")
    if m["mean_actual"] == m["mean_actual"]:
        text_lines.append(f"Mean actual weekly: *{m['mean_actual']:.2f}%*")
    if m["er_bias"] == m["er_bias"]:
        bias = m["er_bias"]
        direction = "under-forecasted" if bias > 0 else "over-forecasted"
        text_lines.append(f"ER bias: *{bias:.2f}%* ({direction} vs reality)")

    text_lines.append("")
    text_lines.append("_Full per-ticker table saved in repo (reports/).*")

    return "\n".join(text_lines)


def run_report_and_notify() -> None:
    info = run_report()
    msg = format_telegram_summary(info)
    send_telegram_message(msg)


if __name__ == "__main__":
    run_report_and_notify()
