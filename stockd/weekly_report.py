# stockd/weekly_report.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stockd import settings
from stockd.telegram_utils import send_telegram_message, send_telegram_document, send_telegram_photo
from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, summarize, dedup_forecasts
from stockd.calibration import load_calibration, apply_calibration


def get_week_bounds(run_date: date) -> tuple[date, date]:
    offset = (run_date.weekday() - 4) % 7
    week_end = run_date - timedelta(days=offset)
    week_start = week_end - timedelta(days=4)
    return week_start, week_end


def _safe_pct(x) -> str:
    try:
        return f"{float(x):+.2f}%"
    except Exception:
        return "n/a"


def _make_chart(df_week: pd.DataFrame, out_path: Path, title: str) -> None:
    if df_week.empty:
        return

    df = df_week.copy()
    df["AbsErrorRaw"] = (df["Realized_Pct"] - df["Model_ER_Pct"]).abs()
    df = df.sort_values("AbsErrorRaw", ascending=False).head(15)

    x = np.arange(len(df))
    width = 0.30

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df["Model_ER_Pct"].values, width, label="Model ER (raw)")
    if "Adj_ER_Pct" in df.columns:
        ax.bar(x, df["Adj_ER_Pct"].values, width, label="Model ER (calibrated)")
    ax.bar(x + width, df["Realized_Pct"].values, width, label="Realized")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Ticker"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("Weekly return (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_weekly_report(run_date: date | None = None) -> dict:
    if run_date is None:
        run_date = date.today()

    week_start, week_end = get_week_bounds(run_date)

    prices = load_prices()
    forecasts = load_forecasts()
    forecasts = dedup_forecasts(forecasts)

    eval_df = evaluate_weekly(prices, forecasts)
    if eval_df.empty:
        return {"week_start": week_start, "week_end": week_end, "excel": None, "png": None, "summary": None}

    # filter pe săptămâna raportată
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end)
    df_week = eval_df[(pd.to_datetime(eval_df["WeekStart"]) == ws) & (pd.to_datetime(eval_df["TargetDate"]) == we)].copy()

    # adaugă calibrare și compară
    calib = load_calibration()
    if not df_week.empty:
        tmp = df_week.rename(columns={"Model_ER_Pct": "ER_Pct"})[["Ticker","Region","ER_Pct"]].copy()
        tmp = apply_calibration(tmp, calib)
        df_week = df_week.merge(tmp[["Ticker","Region","Adj_ER_Pct"]], on=["Ticker","Region"], how="left")

        df_week["Adj_Error_Pct"] = df_week["Realized_Pct"] - df_week["Adj_ER_Pct"]
        df_week["Adj_AbsError_Pct"] = df_week["Adj_Error_Pct"].abs()

        df_week["Raw_DirectionHit"] = df_week["DirectionHit"]
        df_week["Adj_DirectionHit"] = (np.sign(df_week["Adj_ER_Pct"]) == np.sign(df_week["Realized_Pct"])) & (np.sign(df_week["Adj_ER_Pct"]) != 0)

    summary_region = summarize(df_week) if not df_week.empty else pd.DataFrame()

    # total summary
    total = {}
    if not df_week.empty:
        total = {
            "n": int(len(df_week)),
            "hit_raw": float(df_week["Raw_DirectionHit"].mean() * 100.0) if "Raw_DirectionHit" in df_week else float("nan"),
            "hit_adj": float(df_week["Adj_DirectionHit"].mean() * 100.0) if "Adj_DirectionHit" in df_week else float("nan"),
            "mae_raw": float(df_week["AbsError_Pct"].mean()),
            "mae_adj": float(df_week["Adj_AbsError_Pct"].mean()) if "Adj_AbsError_Pct" in df_week else float("nan"),
        }

    # export Excel + PNG
    reports_dir = settings.REPORTS_DIR
    excel_path = reports_dir / f"weekly_bvb_model_vs_real_{week_end.isoformat()}.xlsx"
    png_path = reports_dir / f"weekly_bvb_model_vs_real_{week_end.isoformat()}.png"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_week.to_excel(writer, index=False, sheet_name="Per_Ticker")
        summary_region.to_excel(writer, index=False, sheet_name="Summary_Region")
        pd.DataFrame([total]).to_excel(writer, index=False, sheet_name="Summary_Total")
        pd.DataFrame([calib]).to_excel(writer, index=False, sheet_name="Calibration_Meta")

    _make_chart(
        df_week=df_week,
        out_path=png_path,
        title=f"StockD – Model vs Real ({week_start.isoformat()} → {week_end.isoformat()})",
    )

    # Telegram message
    msg = [
        "*StockD weekly report*",
        f"Week: *{week_start.isoformat()}* → *{week_end.isoformat()}*",
        f"Names: *{total.get('n','n/a')}*",
        f"Hit rate raw: *{total.get('hit_raw', float('nan')):.1f}%*",
        f"Hit rate calibrated: *{total.get('hit_adj', float('nan')):.1f}%*",
        f"MAE raw: *{total.get('mae_raw', float('nan')):.2f}%*",
        f"MAE calibrated: *{total.get('mae_adj', float('nan')):.2f}%*",
        "",
        "_Excel + chart attached._",
    ]

    return {
        "week_start": week_start,
        "week_end": week_end,
        "excel": excel_path,
        "png": png_path,
        "summary_text": "\n".join(msg),
    }


def run_and_notify() -> None:
    info = run_weekly_report()
    if not info.get("excel"):
        print("[REPORT] Nothing to report.")
        return

    try:
        send_telegram_message(info["summary_text"])
    except Exception as exc:
        print(f"[TELEGRAM] Could not send message: {exc}")

    try:
        send_telegram_photo(str(info["png"]), caption=f"Weekly chart {info['week_start']} → {info['week_end']}")
    except Exception as exc:
        print(f"[TELEGRAM] Could not send chart: {exc}")

    try:
        send_telegram_document(str(info["excel"]), caption=f"Weekly report {info['week_start']} → {info['week_end']}")
    except Exception as exc:
        print(f"[TELEGRAM] Could not send Excel: {exc}")


if __name__ == "__main__":
    run_and_notify()
