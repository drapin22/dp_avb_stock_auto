# stockd/learning.py
from __future__ import annotations

import argparse

from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, summarize
from stockd.calibration import build_region_calibration, save_calibration


def run_learning() -> None:
    prices = load_prices()
    forecasts = load_forecasts()

    eval_df = evaluate_weekly(prices, forecasts)
    if eval_df.empty:
        print("[LEARN] No evaluable forecasts yet.")
        return

    summary_df = summarize(eval_df)

    # save eval outputs
    from stockd import settings
    eval_df.to_csv(settings.MODEL_EVAL_DETAILED, index=False)
    summary_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)
    print(f"[LEARN] Saved eval detailed: {settings.MODEL_EVAL_DETAILED}")
    print(f"[LEARN] Saved eval summary: {settings.MODEL_EVAL_SUMMARY}")

    calib = build_region_calibration(eval_df)
    save_calibration(calib)
    print(f"[LEARN] Saved calibration: {settings.CALIBRATION_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", default=True)
    args = parser.parse_args()
    run_learning()
