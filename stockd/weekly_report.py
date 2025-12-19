from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from stockd import settings
from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, summarize, dedup_forecasts
from stockd.telegram_utils import send_chunked_message, send_telegram_document, send_telegram_photo


def _latest_forecast_window(forecasts: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    f = forecasts.copy()
    f["WeekStart"] = pd.to_datetime(f["WeekStart"], errors="coerce")
    f["TargetDate"] = pd.to_datetime(f["TargetDate"], errors="coerce")
    f = f.dropna(subset=["WeekStart", "TargetDate"])

    if f.empty:
        raise ValueError("No valid WeekStart/TargetDate in forecasts.")

    # IMPORTANT: ignore future TargetDates (weekly report must compare vs realized)
    today = pd.Timestamp(date.today())
    completed = f[f["TargetDate"] < today]
    if completed.empty:
        raise ValueError("No completed TargetDate yet (all forecasts are in the future).")

    last_target = completed["TargetDate"].max()
    row = completed.loc[completed["TargetDate"] == last_target].iloc[0]
    return pd.to_datetime(row["WeekStart"]), pd.to_datetime(row["TargetDate"])


def _plot_error_bars(eval_df: pd.DataFrame, out_png: Path) -> None:
    d = eval_df.sort_values("AbsError_Pct", ascending=False).head(20).copy()
    if d.empty:
        return

    labels = [f"{t}({r})" for t, r in zip(d["Ticker"], d["Region"])]
    errs = d["Error_Pct"].fillna(0.0).astype(float).tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, errs)
    plt.xticks(rotation=60, ha="right")
    plt.axhline(0, linewidth=1)
    plt.title("Top 20 forecast errors (Model - Real) [%]")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _write_excel(eval_df: pd.DataFrame, summary_df: pd.DataFrame, xlsx_path: Path) -> None:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        eval_df.to_excel(writer, sheet_name="Detailed", index=False)


def run_weekly_report() -> Dict:
    prices = load_prices()
    forecasts = load_forecasts()
    forecasts = dedup_forecasts(forecasts)

    if prices.empty:
        return {"ok": False, "error": "prices_history.csv is empty/missing"}
    if forecasts.empty:
        return {"ok": False, "error": "forecasts_stockd.csv is empty/missing"}

    try:
        ws, we = _latest_forecast_window(forecasts)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    f_week = forecasts[(pd.to_datetime(forecasts["WeekStart"]) == ws) & (pd.to_datetime(forecasts["TargetDate"]) == we)].copy()
    if f_week.empty:
        return {"ok": False, "error": f"No forecasts found for window {ws.date()}..{we.date()}"}

    eval_df = evaluate_weekly(prices, f_week)
    if eval_df.empty:
        return {"ok": False, "error": f"No evaluation rows (missing prices for {ws.date()}..{we.date()})"}

    summary_df = summarize(eval_df)

    # Outputs
    eval_csv = settings.REPORTS_DIR / f"model_vs_real_{ws.date()}_{we.date()}.csv"
    summary_csv = settings.REPORTS_DIR / f"model_vs_real_summary_{ws.date()}_{we.date()}.csv"
    report_png = settings.REPORTS_DIR / f"weekly_report_{we.date()}.png"
    report_xlsx = settings.REPORTS_DIR / f"weekly_bvb_model_vs_real_{we.date()}.xlsx"

    eval_df.to_csv(eval_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_error_bars(eval_df, report_png)
    _write_excel(eval_df, summary_df, report_xlsx)

    return {
        "ok": True,
        "week_start": str(ws.date()),
        "week_end": str(we.date()),
        "n": int(eval_df.shape[0]),
        "mae": float(eval_df["AbsError_Pct"].mean()),
        "hitrate": float(eval_df["DirectionHit"].mean()),
        "paths": {
            "eval_csv": str(eval_csv),
            "summary_csv": str(summary_csv),
            "png": str(report_png),
            "xlsx": str(report_xlsx),
        },
    }


def format_telegram_summary(info: Dict) -> str:
    if not info.get("ok"):
        return f"Weekly report failed: {info.get('error')}"

    return (
        "StockD weekly report (Model vs Real)\n"
        f"Window: {info['week_start']} → {info['week_end']}\n"
        f"Tickers: {info['n']}\n"
        f"MAE: {info['mae']:.2f}%\n"
        f"Hit rate: {info['hitrate']*100:.1f}%\n"
    )


def run_and_notify() -> None:
    info = run_weekly_report()
    msg = format_telegram_summary(info)
    send_chunked_message(msg)

    if info.get("ok"):
        p = info["paths"]
        send_telegram_photo(p["png"], caption="Top errors (chart)")
        send_telegram_document(p["xlsx"], caption="Weekly report (Excel)")
        send_telegram_document(p["eval_csv"], caption="Model vs Real (CSV)")
        send_telegram_document(p["summary_csv"], caption="Summary (CSV)")


if __name__ == "__main__":
    run_and_notify()
