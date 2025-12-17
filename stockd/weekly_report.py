# stockd/weekly_report.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from stockd import settings
from stockd.telegram_utils import send_telegram_document, send_telegram_photo


# ------------ Utilitare de dată / fișiere -----------------


def get_week_bounds(run_date: date) -> tuple[date, date]:
    """
    Returnează (week_start, week_end) pentru săptămâna de raport.
    Folosim convenția:
    - Dacă rulezi sâmbătă/duminică -> săptămâna anterioară (lun-vin)
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


# ------------ Loaders -----------------


def load_holdings_ro() -> pd.DataFrame:
    """
    În repo-ul tău, holdings RO sunt definite în settings.HOLDINGS_RO
    """
    rows = []
    for t in settings.HOLDINGS_RO:
        rows.append({"Ticker": t, "Region": "RO"})
    return pd.DataFrame(rows)


def load_forecasts_for_week(week_start: date, week_end: date) -> pd.DataFrame:
    """
    Citește forecasts_stockd.csv și ia forecasturile pentru săptămâna curentă (WeekStart=week_start, TargetDate=week_end).
    Păstrează ultima predicție per Ticker/Region (dacă ai duplicat pe Date).
    """
    fc_path = Path(settings.FORECASTS_CSV)
    if not fc_path.exists():
        print(f"[REPORT] Missing forecasts file: {fc_path}")
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    df = pd.read_csv(fc_path, parse_dates=["Date", "WeekStart", "TargetDate"])
    df = df[(df["WeekStart"].dt.date == week_start) & (df["TargetDate"].dt.date == week_end)]

    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    # păstrează ultima predicție pentru combinația aceea
    df = (
        df.sort_values("Date")
          .groupby(["WeekStart", "TargetDate", "Ticker", "Region"], as_index=False)
          .tail(1)
    )

    return df[["Ticker", "Region", "ER_Pct"]].copy()


def load_prices_for_week(week_start: date, week_end: date) -> pd.DataFrame:
    """
    În repo ai deja un fișier de prețuri unificat (sau un path în settings).
    Dacă ai un alt path, ajustează settings.PRICES_ALL_CSV.
    """
    px_path = Path(getattr(settings, "PRICES_ALL_CSV", "data/prices_all.csv"))
    if not px_path.exists():
        print(f"[REPORT] Missing prices file: {px_path}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Close"])

    df = pd.read_csv(px_path, parse_dates=["Date"])
    df = df[(df["Date"].dt.date >= week_start) & (df["Date"].dt.date <= week_end)].copy()
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


# ------------ Model vs Real table -----------------


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

    # păstrăm datele din săptămână și pivotăm în coloane Mon..Fri
    prices = prices.copy()
    prices["DOW"] = prices["Date"].dt.day_name().str[:3]  # Mon, Tue, Wed...
    pivot = prices.pivot_table(
        index=["Ticker", "Region"],
        columns="DOW",
        values="Close",
        aggfunc="last",
    )

    # Redenumim în Mon/Tue/...
    pivot.columns = [f"{c}_Close" for c in pivot.columns]
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

    # asigurăm coloane Mon..Fri chiar dacă lipsesc
    for c in ["Mon_Close", "Tue_Close", "Wed_Close", "Thu_Close", "Fri_Close"]:
        if c not in merged.columns:
            merged[c] = np.nan

    # ordonare
    cols = [
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
    return merged[cols].sort_values(["Region", "Ticker"])


def portfolio_metrics(weekly_table: pd.DataFrame) -> dict:
    if weekly_table is None or weekly_table.empty:
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


def save_weekly_chart(
    weekly_table: pd.DataFrame,
    out_dir: Path,
    week_start: date,
    week_end: date,
    top_n: int = 15,
) -> Path | None:
    """
    Salvează un grafic PNG cu Model_ER_Pct vs Actual_Weekly_Pct.

    Arată top_n tickere cu cea mai mare eroare absolută (ca să fie lizibil).
    """
    if weekly_table is None or weekly_table.empty:
        return None

    needed = {"Ticker", "Model_ER_Pct", "Actual_Weekly_Pct"}
    if not needed.issubset(set(weekly_table.columns)):
        print("[REPORT] Missing columns for chart. Skipping chart.")
        return None

    df = weekly_table[["Ticker", "Model_ER_Pct", "Actual_Weekly_Pct"]].copy()
    df["AbsError"] = (df["Actual_Weekly_Pct"] - df["Model_ER_Pct"]).abs()
    df = df.sort_values("AbsError", ascending=False).head(top_n)

    tickers = df["Ticker"].astype(str).tolist()
    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df["Model_ER_Pct"].values, width, label="Model ER (%)")
    ax.bar(x + width / 2, df["Actual_Weekly_Pct"].values, width, label="Real (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Return weekly (%)")
    ax.set_title(f"BVB – Model vs Real ({week_start.isoformat()} → {week_end.isoformat()})")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / f"weekly_bvb_model_vs_real_{week_end.isoformat()}.png"
    fig.savefig(chart_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[REPORT] Saved chart to {chart_path}")
    return chart_path


# ------------ Telegram -----------------


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Skipping.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        print("[TELEGRAM] Message sent.")
    except Exception as exc:
        print(f"[TELEGRAM] Exception: {exc}")


# ------------ Main orchestration -----------------


def run_report(run_date: date | None = None) -> dict:
    if run_date is None:
        run_date = date.today()

    week_start, week_end = get_week_bounds(run_date)
    print(f"[REPORT] Week = {week_start} -> {week_end}")

    # Load data
    holdings_ro = load_holdings_ro()
    forecasts = load_forecasts_for_week(week_start, week_end)
    prices = load_prices_for_week(week_start, week_end)

    # Keep only holdings tickers (RO) + whatever forecast tickers exist for that week
    tickers = set(holdings_ro["Ticker"].astype(str).tolist())
    tickers |= set(forecasts["Ticker"].astype(str).tolist())

    if not prices.empty:
        prices = prices[prices["Ticker"].astype(str).isin(tickers)].copy()

    # Weekly table
    weekly_table = build_weekly_table(prices, forecasts, week_start, week_end)
    metrics = portfolio_metrics(weekly_table)

    # Save Excel
    reports_dir = get_reports_dir()
    out_path = reports_dir / f"weekly_bvb_model_vs_real_{week_end.isoformat()}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        weekly_table.to_excel(writer, index=False, sheet_name="Per_Ticker")

        summary_df = pd.DataFrame(
            {
                "Metric": ["n_names", "hit_rate", "mean_model_er", "mean_actual", "er_bias"],
                "Value": [
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
        text_lines.append(f"Directional hit rate: *{hit_rate_pct:.1f}%*")
    else:
        text_lines.append("Directional hit rate: _n/a_")

    if m["mean_model_er"] == m["mean_model_er"]:
        text_lines.append(f"Avg model ER: *{m['mean_model_er']:.2f}%*")
    else:
        text_lines.append("Avg model ER: _n/a_")

    if m["mean_actual"] == m["mean_actual"]:
        text_lines.append(f"Avg actual weekly: *{m['mean_actual']:.2f}%*")
    else:
        text_lines.append("Avg actual weekly: _n/a_")

    if m["er_bias"] == m["er_bias"]:
        bias = m["er_bias"]
        bias_txt = f"{bias:+.2f}%"
        text_lines.append(f"ER bias (actual - model): *{bias_txt}*")
    else:
        text_lines.append("ER bias: _n/a_")

    text_lines.append("")
    text_lines.append("_Excel report attached in this chat (plus chart)._")

    return "\n".join(text_lines)


def run_report_and_notify() -> None:
    info = run_report()
    msg = format_telegram_summary(info)

    # 1) mesaj text
    send_telegram_message(msg)

    # 2) grafic PNG (Model vs Real)
    try:
        chart_path = save_weekly_chart(
            weekly_table=info["weekly_table"],
            out_dir=Path(info["excel_path"]).parent,
            week_start=info["week_start"],
            week_end=info["week_end"],
        )
        if chart_path is not None:
            send_telegram_photo(
                str(chart_path),
                caption=f"BVB model vs real – {info['week_start'].isoformat()} → {info['week_end'].isoformat()}",
            )
    except Exception as exc:
        print(f"[TELEGRAM] Could not send chart: {exc}")

    # 3) Excel complet
    try:
        send_telegram_document(
            str(info["excel_path"]),
            caption=f"Weekly BVB model vs real – {info['week_start'].isoformat()} → {info['week_end'].isoformat()}",
        )
    except Exception as exc:
        print(f"[TELEGRAM] Could not send Excel: {exc}")


if __name__ == "__main__":
    run_report_and_notify()
