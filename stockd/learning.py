# stockd/learning.py
from __future__ import annotations

import pandas as pd

from stockd import settings
from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, summarize
from stockd.calibration import load_calibration, save_calibration, build_region_calibration, merge_coach_suggestions
from stockd.llm_coach import coach_calibration_suggestions


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

    # important: scriem calibrarea chiar dacă nu avem ce evalua
    calib = load_calibration()
    calib["notes"] = f"no eval this run: {reason}"
    save_calibration(calib)

    print(f"[LEARN] No evaluable rows. Wrote empty eval files and kept calibration. Reason: {reason}")


def run_learning(use_coach: bool = True) -> None:
    prices = load_prices()
    forecasts = load_forecasts()

    print(f"[LEARN] prices rows={len(prices)} file={settings.PRICES_HISTORY}")
    print(f"[LEARN] forecasts rows={len(forecasts)} file={settings.FORECASTS_FILE}")

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
    print(f"[LEARN] Saved eval detailed: {settings.MODEL_EVAL_DETAILED} rows={len(eval_df)}")
    print(f"[LEARN] Saved eval summary: {settings.MODEL_EVAL_SUMMARY} rows={len(summary_df)}")

    # 1) calibrare deterministă
    calib = build_region_calibration(eval_df)

    # 2) coach (opțional): propune clip/alpha/caps
    if use_coach:
        worst = (
            eval_df.sort_values("AbsError_Pct", ascending=False)
            .head(20)[["Ticker","Region","Model_ER_Pct","Realized_Pct","Error_Pct","AbsError_Pct","DirectionHit","WeekStart","TargetDate"]]
        )

        current = load_calibration()
        coach = coach_calibration_suggestions(summary_df, worst, current)
        calib = merge_coach_suggestions(calib, coach)

        print(f"[LEARN] coach_status={coach.get('_coach_status')}")

    save_calibration(calib)
    print(f"[LEARN] Saved calibration: {settings.CALIBRATION_FILE}")


if __name__ == "__main__":
    run_learning()
