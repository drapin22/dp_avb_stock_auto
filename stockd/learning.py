# stockd/learning.py
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd

from stockd import settings
from stockd.evaluation import evaluate_weekly, summarize
from stockd.scoring import compute_scores, save_scores
from stockd.calibration import build_region_calibration, save_calibration
from stockd.mentor import run_mentor
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def _last_completed_week_end(run_date: date) -> date:
    # ultima vineri înainte de run_date (workflow rulează sâmbăta)
    offset = (run_date.weekday() - 4) % 7
    return run_date - timedelta(days=offset)


def run_learning() -> None:
    eval_df = evaluate_weekly()
    if eval_df.empty:
        send_telegram_message("StockD learning: no completed weeks available for evaluation.")
        return

    summary_df = summarize(eval_df)

    # salvează (overwrite) fișierele de referință
    eval_df.to_csv(settings.MODEL_EVAL_DETAILED, index=False)
    summary_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    # scoring
    scores_df = compute_scores(eval_df)
    save_scores(scores_df)

    # calibrare deterministă
    calib = build_region_calibration(eval_df)
    save_calibration(calib)

    # mentor pe ultima săptămână completă
    week_end = _last_completed_week_end(date.today())
    last_week = eval_df[eval_df["TargetDate"].dt.date == week_end].copy()
    mentor_info = run_mentor(last_week, week_end=week_end)

    # Telegram (scurt + atașamente)
    msg_lines = []
    msg_lines.append("StockD learning update")
    msg_lines.append(f"Eval rows: {len(eval_df)}")
    if not summary_df.empty:
        recent = summary_df.sort_values("WeekStart").groupby("Region").tail(1)
        for _, r in recent.iterrows():
            msg_lines.append(
                f"{r['Region']}: n={int(r['n'])}, hit={float(r['hit_rate']):.2f}, "
                f"MAE={float(r['mean_abs_error']):.2f}pp, bias={float(r['mean_error']):+.2f}pp"
            )
    msg_lines.append(f"Mentor: {mentor_info.get('status')}")

    send_telegram_message("\n".join(msg_lines))
    send_telegram_document(str(settings.MODEL_EVAL_SUMMARY), caption="Model eval summary")
    send_telegram_document(str(settings.SCORES_FILE), caption="Ticker reliability scores")
    send_telegram_document(str(settings.CALIBRATION_FILE), caption="Calibration")
    if settings.MENTOR_OVERRIDES_FILE.exists():
        send_telegram_document(str(settings.MENTOR_OVERRIDES_FILE), caption="Mentor overrides")
    if mentor_info.get("md_path"):
        send_telegram_document(str(mentor_info["md_path"]), caption="Mentor postmortem")


if __name__ == "__main__":
    run_learning()
