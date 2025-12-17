# stockd/weekly_report.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stockd import settings
from stockd.telegram_utils import (
    send_telegram_message,
    send_telegram_document,
    send_telegram_photo,
)


# ---------------------------
# Date helpers
# ---------------------------

def get_week_bounds(run_date: date) -> tuple[date, date]:
    """
    Returnează (week_start, week_end) pentru săptămâna de raport (luni -> vineri).

    Convenție:
    - Pentru orice zi de rulare, raportăm ultima săptămână completă care se termină vineri.
    Exemplu:
      - dacă rulezi miercuri 17 dec -> week_end = vineri 12 dec, week_start = luni 8 dec
    """
    offset = (run_date.weekday() - 4) % 7  # 4 = Friday
    week_end = run_date - timedelta(days=offset)
    week_start = week_end - timedelta(days=4)
    return week_start, week_end


def get_reports_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# ---------------------------
# Loaders
# ---------------------------

def load_holdings_ro() -> pd.DataFrame:
    """
    Acceptă 2 forme pentru settings.HOLDINGS_RO:
      1) list/tuple/set de tickere (ex: ["TLV", "SNP"])
      2) Path/str către fișier cu tickere:
         - CSV cu coloană 'Ticker' (recomandat) sau 'ticker'
         - TXT cu un ticker pe linie

    Returnează DataFrame cu coloane: Ticker, Region
    """
    src = getattr(settings, "HOLDINGS_RO", None)

    if src is None:
        return pd.DataFrame(columns=["Ticker", "Region"])

    # Case 1: listă/tuplu/set
    if isinstance(src, (list, tuple, set)):
        tickers = [str(t).strip() for t in src if str(t).strip()]
        return pd.DataFrame({"Ticker": tickers, "Region": "RO"})

    # Case 2: Path sau str către fișier
    path = Path(src)
    if not path.exists():
        print(f"[REPORT] HOLDINGS_RO points to missing file: {path}")
        return pd.DataFrame(columns=["Ticker", "Region"])

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "Ticker" if "Ticker" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if col is None:
            raise ValueError(f"HOLDINGS_RO file {path} must have a 'Ticker' (or 'ticker') column.")
        tickers = df[col].astype(str).str.strip()
        tickers = tickers[tickers != ""]
        return pd.DataFrame({"Ticker": tickers.tolist(), "Region": "RO"})

    # Fallback: txt cu un ticker pe linie
    lines = path.read_text(encoding="utf-8").splitlines()
    tickers = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return pd.DataFrame({"Ticker": tickers, "Region": "RO"})


def load_forecasts_for_week(week_start: date, week_end: date) -> pd.DataFrame:
    """
    Citește forecasts_stockd.csv și ia forecasturile pentru săptămâna curentă (WeekStart, TargetDate).
    Păstrează ultima predicție per Ticker/Region (dacă ai duplicat pe Date).

    Path:
      - settings.FORECASTS_CSV dacă există
      - altfel fallback: data/forecasts_stockd.csv
    """
    fc_path = Path(getattr(settings, "FORECASTS_CSV", "data/forecasts_stockd.csv"))
    if not fc_path.exists():
        print(f"[REPORT] Missing forecasts file: {fc_path}")
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    df = pd.read_csv(fc_path, parse_dates=["Date", "WeekStart", "TargetDate"])

    # filtrare pe săptămâna raportată
    df = df[(df["WeekStart"].dt.date == week_start) & (df["TargetDate"].dt.date == week_end)]
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    # păstrează ultima predicție pentru combinația respectivă
    df = (
        df.sort_values("Date")
          .groupby(["WeekStart", "TargetDate", "Ticker", "Region"], as_index=False)
          .tail(1)
    )

    return df[["Ticker", "Region", "ER_Pct"]].copy()


def load_prices_for_week(week_start: date, week_end: date) -> pd.DataFrame:
    """
    Încarcă prețurile dintr-un fișier unificat.

    Path:
      - settings.PRICES_ALL_CSV dacă există
      - altfel fallback: data/prices_all.csv
    """
    px_path = Path(getattr(settings, "PRICES_ALL_CSV", "data/prices_all.csv"))
    if not px_path.exists():
        print(f"[REPORT] Missing prices file: {px_path}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Close"])

    df = pd.read_csv(px_path, parse_dates=["Date"])
    df = df[(df["Date"].dt.date >= week_start) & (df["Date"].dt.date <= week_end)].copy()
    df.sort_values(["Ticker", "Region", "Date"], inplace=True)
    return df


# ---------------------------
# Build weekly table
# ---------------------------

def build_weekly_table(
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creează tabelul pe tickere cu:
    Mon..Fri close, Actual_Weekly_Pct, Model_ER_Pct, Verdict
    """
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "Ticker", "Region",
                "Mon_Close", "Tue_Close", "Wed_Close", "Thu_Close", "Fri_Close",
                "Actual_Weekly_Pct", "Model_ER_Pct", "Verdict",
            ]
        )

    p = prices.copy()
    p["DOW"] = p["Date"].dt.day_name().str[:3]  # Mon, Tue, Wed...

    pivot = p.pivot_table(
        index=["Ticker", "Region"],
        columns="DOW",
        values="Close",
        aggfunc="last",
    )
    pivot.columns = [f"{c}_Close" for c in pivot.columns]
    pivot.reset_index(inplace=True)

    # Actual weekly return: last/first în săptămână per (Ticker, Region)
    p_sorted = p.sort_values(["Ticker", "Region", "Date"])
    first = p_sorted.groupby(["Ticker", "Region"])["Close"].first()
    last = p_sorted.groupby(["Ticker", "Region"])["Close"].last()
    actual_weekly = (last / first - 1.0) * 100.0
    actual_df = actual_weekly.rename("Actual_Weekly_Pct").reset_index()

    merged = pivot.merge(actual_df, on=["Ticker", "Region"], how="left")
    merged = merged.merge(forecasts.rename(columns={"ER_Pct": "Model_ER_Pct"}), on=["Ticker", "Region"], how="left")

    def verdict_row(row):
        a = row.get("Actual_Weekly_Pct", np.nan)
        m = row.get("Model_ER_Pct", np.nan)
        if pd.isna(a) or pd.isna(m):
            return "NO_DATA"
        if a == 0 or m == 0:
            return "NEUTRAL"
        return "HIT" if np.sign(a) == np.sign(m) else "MISS"

    merged["Verdict"] = merged.apply(verdict_row, axis=1)

    for c in ["Mon_Close", "Tue_Close", "Wed_Close", "Thu_Close", "Fri_Close"]:
        if c not in merged.columns:
            merged[c] = np.nan

    cols = [
        "Ticker", "Region",
        "Mon_Close", "Tue_Close", "Wed_Close", "Thu_Close", "Fri_Close",
        "Actual_Weekly_Pct", "Model_ER_Pct", "Verdict",
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

    valid = weekly_table[weekly_table["Verdict"].isin(["HIT", "MISS"])].copy()
    n = len(valid)
    hits = int((valid["Verdict"] == "HIT").sum())

    mean_model = valid["Model_ER_Pct"].mean()
    mean_actual = valid["Actual_Weekly_Pct"].mean()
    er_bias = mean_actual - mean_model

    return {
        "n_names": int(n),
        "hit_rate": float(hits / n) if n > 0 else np.nan,
        "mean_model_er": float(mean_model) if n > 0 else np.nan,
        "mean_actual": float(mean_actual) if n > 0 else np.nan,
        "er_bias": float(er_bias) if n > 0 else np.nan,
    }


# ---------------------------
# Chart + Excel output
# ---------------------------

def save_weekly_chart(
    weekly_table: pd.DataFrame,
    out_dir: Path,
    week_start: date,
    week_end: date,
    top_n: int = 15,
) -> Path | None:
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


def save_excel_report(
    weekly_table: pd.DataFrame,
    metrics: dict,
    out_path: Path,
) -> None:
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


# ---------------------------
# Orchestration
# ---------------------------

def run_report(run_date: date | None = None) -> dict:
    if run_date is None:
        run_date = date.today()

    week_start, week_end = get_week_bounds(run_date)
    print(f"[REPORT] Week = {week_start.isoformat()} -> {week_end.isoformat()}")

    holdings_ro = load_holdings_ro()
    forecasts = load_forecasts_for_week(week_start, week_end)
    prices = load_prices_for_week(week_start, week_end)

    tickers = set(holdings_ro["Ticker"].astype(str).tolist()) | set(forecasts["Ticker"].astype(str).tolist())

    if not prices.empty:
        prices = prices[prices["Ticker"].astype(str).isin(tickers)].copy()

    weekly_table = build_weekly_table(prices, forecasts)
    metrics = portfolio_metrics(weekly_table)

    reports_dir = get_reports_dir()
    excel_path = reports_dir / f"weekly_bvb_model_vs_real_{week_end.isoformat()}.xlsx"
    save_excel_report(weekly_table, metrics, excel_path)
    print(f"[REPORT] Saved Excel report to {excel_path}")

    return {
        "week_start": week_start,
        "week_end": week_end,
        "weekly_table": weekly_table,
        "metrics": metrics,
        "excel_path": excel_path,
    }


def format_telegram_summary(info: dict) -> str:
    ws: date = info["week_start"]
    we: date = info["week_end"]
    m: dict = info["metrics"]

    def is_nan(x):
        return x != x  # NaN trick

    lines = [
        "*StockD – Weekly BVB Model vs Market*",
        f"Week: *{ws.isoformat()}* → *{we.isoformat()}*",
        "",
        f"Names evaluated: *{m['n_names']}*",
    ]

    if not is_nan(m["hit_rate"]):
        lines.append(f"Directional hit rate: *{m['hit_rate'] * 100:.1f}%*")
    else:
        lines.append("Directional hit rate: _n/a_")

    if not is_nan(m["mean_model_er"]):
        lines.append(f"Avg model ER: *{m['mean_model_er']:.2f}%*")
    else:
        lines.append("Avg model ER: _n/a_")

    if not is_nan(m["mean_actual"]):
        lines.append(f"Avg actual weekly: *{m['mean_actual']:.2f}%*")
    else:
        lines.append("Avg actual weekly: _n/a_")

    if not is_nan(m["er_bias"]):
        lines.append(f"ER bias (actual - model): *{m['er_bias']:+.2f}%*")
    else:
        lines.append("ER bias: _n/a_")

    lines.append("")
    lines.append("_Excel report + chart attached._")
    return "\n".join(lines)


def run_report_and_notify() -> None:
    info = run_report()
    msg = format_telegram_summary(info)

    # 1) text
    try:
        send_telegram_message(msg)
    except Exception as exc:
        print(f"[TELEGRAM] Could not send message: {exc}")

    # 2) chart
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

    # 3) excel
    try:
        send_telegram_document(
            str(info["excel_path"]),
            caption=f"Weekly BVB model vs real – {info['week_start'].isoformat()} → {info['week_end'].isoformat()}",
        )
    except Exception as exc:
        print(f"[TELEGRAM] Could not send Excel: {exc}")


if __name__ == "__main__":
    run_report_and_notify()
