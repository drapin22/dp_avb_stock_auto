from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from stockd import settings
from stockd.evaluation import load_forecasts, load_prices, dedup_forecasts, evaluate_weekly, summarize
from stockd.telegram_utils import send_telegram_document, send_telegram_message, send_telegram_photo


def _week_from_latest_forecast(forecasts: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    f = forecasts.dropna(subset=["WeekStart", "TargetDate"]).copy()
    last_target = f["TargetDate"].max()
    row = f.loc[f["TargetDate"] == last_target].iloc[0]
    return pd.Timestamp(row["WeekStart"]).normalize(), pd.Timestamp(row["TargetDate"]).normalize()


def _plot_bar(eval_df: pd.DataFrame, out_png: Path, title: str) -> None:
    df = eval_df.sort_values("RealizedReturnPct", ascending=False).copy()
    labels = [f"{t}" for t in df["Ticker"].tolist()]
    x = range(len(df))

    plt.figure(figsize=(12, 5))
    plt.bar(x, df["PredictedReturnPct"], label="Model (raw)")
    plt.bar(x, df["RealizedReturnPct"], label="Real")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def run_weekly_report() -> Dict:
    prices = load_prices()
    forecasts = dedup_forecasts(load_forecasts())

    if forecasts.empty or prices.empty:
        return {"status": "EMPTY", "message": "Missing prices or forecasts."}

    week_start, week_end = _week_from_latest_forecast(forecasts)
    f_week = forecasts[(forecasts["WeekStart"] == week_start) & (forecasts["TargetDate"] == week_end)].copy()
    if f_week.empty:
        return {"status": "EMPTY", "message": "No forecasts found for latest week."}

    eval_df = evaluate_weekly(prices, f_week)
    if eval_df.empty:
        return {"status": "EMPTY", "message": "Evaluation produced no rows."}

    settings.MODEL_EVAL_DETAILED.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(settings.MODEL_EVAL_DETAILED, index=False)

    summ = summarize(eval_df)
    summ_rows = [{"Region": region, **m} for region, m in summ.items()]
    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    xlsx_path = settings.REPORTS_DIR / f"weekly_bvb_model_vs_real_{week_end.date()}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        eval_df.to_excel(writer, index=False, sheet_name="Eval")
        summ_df.to_excel(writer, index=False, sheet_name="Summary")

    png_path = settings.REPORTS_DIR / f"weekly_chart_{week_start.date()}_{week_end.date()}.png"
    _plot_bar(eval_df, png_path, title=f"StockD - Model vs Real ({week_start.date()} → {week_end.date()})")

    overall_hit = float(eval_df["hit"].mean()) if len(eval_df) else 0.0
    overall_mae = float(eval_df["abs_error_pp"].mean()) if len(eval_df) else 0.0

    msg_lines = [
        "StockD weekly report",
        f"Week: {week_start.date()} → {week_end.date()}",
        f"Names: {len(eval_df)}",
        f"Hit rate: {overall_hit*100:.1f}%",
        f"MAE: {overall_mae:.2f}pp",
        "",
        "Per region:",
    ]
    for region, m in summ.items():
        msg_lines.append(f"{region}: n={int(m['n'])}, hit={m['hit']:.2f}, MAE={m['MAE_pp']:.2f}pp, bias={m['bias_pp']:.2f}pp")

    send_telegram_message("\n".join(msg_lines))
    send_telegram_photo(png_path, caption=f"Weekly chart {week_start.date()} → {week_end.date()}")
    send_telegram_document(xlsx_path, caption=f"Weekly report {week_start.date()} → {week_end.date()}")

    return {
        "status": "OK",
        "week_start": str(week_start.date()),
        "week_end": str(week_end.date()),
        "xlsx": str(xlsx_path),
        "png": str(png_path),
        "eval_rows": int(len(eval_df)),
    }


def run_and_notify() -> None:
    info = run_weekly_report()
    if info.get("status") != "OK":
        send_telegram_message(f"Weekly report: {info.get('status')} {info.get('message','')}")


if __name__ == "__main__":
    run_and_notify()
