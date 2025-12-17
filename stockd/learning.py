# stockd/learning.py
from __future__ import annotations

import pandas as pd

from stockd import settings
from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, summarize
from stockd.calibration import load_calibration, save_calibration, build_region_calibration
from stockd.scoring import compute_scores, save_scores
from stockd.mentor import run_mentor_diagnostics
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def _write_empty_eval_outputs(reason: str) -> None:
    empty_eval = pd.DataFrame(columns=[
        "WeekStart","TargetDate","Date","ModelVersion","Ticker","Region","HorizonDays",
        "Model_ER_Pct","CloseStart","CloseEnd","Realized_Pct","Error_Pct","AbsError_Pct","DirectionHit","Notes"
    ])
    empty_sum = pd.DataFrame(columns=[
        "WeekStart","ModelVersion","Region","n","hit_rate","mean_abs_error","mean_error","median_abs_error"
    ])

    empty_eval.to_csv(settings.MODEL_EVAL_DETAILED, index=False)
    empty_sum.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    calib = load_calibration()
    calib["notes"] = f"no eval this run: {reason}"
    save_calibration(calib)

    send_telegram_message(f"StockD learning: no evaluable rows. Reason: {reason}")


def run_learning(use_mentor: bool = True) -> None:
    prices = load_prices()
    forecasts = load_forecasts()

    if forecasts.empty or prices.empty:
        _write_empty_eval_outputs("forecasts empty or prices empty")
        return

    eval_df = evaluate_weekly(prices, forecasts)
    if eval_df is None or eval_df.empty:
        _write_empty_eval_outputs("evaluate_weekly produced empty")
        return

    summary_df = summarize(eval_df)

    eval_df.to_csv(settings.MODEL_EVAL_DETAILED, index=False)
    summary_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    # 1) scoring
    scores_df = compute_scores(eval_df)
    scores_path = save_scores(scores_df)

    # 2) calibrare deterministă
    calib = build_region_calibration(eval_df)
    save_calibration(calib)

    # 3) mentor diagnostics + overrides safe (opțional)
    mentor_info = {"status": "SKIPPED"}
    if use_mentor:
        mentor_info = run_mentor_diagnostics(eval_df)

    # Telegram summary
    try:
        msg = []
        msg.append("*StockD learning update*")
        msg.append(f"Eval rows: *{len(eval_df)}*")
        if not summary_df.empty:
            # sumar pe regiuni
            for _, r in summary_df.iterrows():
                msg.append(
                    f"{r['Region']}: n={int(r['n'])}, hit={float(r['hit_rate']):.2f}, "
                    f"MAE={float(r['mean_abs_error']):.2f}pp, bias={float(r['mean_error']):+.2f}pp"
                )
        msg.append("")
        msg.append(f"Calibration updated: `{settings.CALIBRATION_FILE}`")
        msg.append(f"Scores updated: `{scores_path}`")
        msg.append(f"Mentor: `{mentor_info.get('status')}` (overrides: `{mentor_info.get('overrides_path','')}`)")

        send_telegram_message("\n".join(msg))

        send_telegram_document(str(settings.MODEL_EVAL_SUMMARY), caption="Weekly evaluation summary")
        send_telegram_document(str(scores_path), caption="Ticker reliability scores")
        if mentor_info.get("overrides_path"):
            send_telegram_document(str(mentor_info["overrides_path"]), caption="Mentor overrides (safe guardrails)")
    except Exception as exc:
        print(f"[LEARN] Telegram notify failed: {exc}")


if __name__ == "__main__":
    run_learning()
