from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from stockd import settings
from stockd.evaluation import load_prices, load_forecasts, dedup_forecasts, evaluate_weekly, summarize
from stockd.telegram_utils import send_telegram_document, send_telegram_message
from stockd.mentor import run_mentor


def _jsonable(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_learning() -> None:
    prices = load_prices()
    forecasts = dedup_forecasts(load_forecasts())

    if prices.empty or forecasts.empty:
        send_telegram_message("StockD learning: missing prices or forecasts.")
        return

    last_target = forecasts["TargetDate"].max()
    f_week = forecasts[forecasts["TargetDate"] == last_target].copy()

    eval_df = evaluate_weekly(prices, f_week)
    if eval_df.empty:
        send_telegram_message("StockD learning: evaluation produced no rows.")
        return

    summ = summarize(eval_df)
    rows = [{"Region": r, **m} for r, m in summ.items()]
    summary_df = pd.DataFrame(rows)

    settings.MODEL_EVAL_DETAILED.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(settings.MODEL_EVAL_DETAILED, index=False)
    summary_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    overall_bias = float(eval_df["bias_pp"].mean())
    global_multiplier = float(max(0.6, min(1.4, 1.0 - overall_bias / 20.0)))

    calib = {
        "week_end": str(pd.Timestamp(last_target).date()),
        "global_multiplier": global_multiplier,
        "clip_pct": 8.0,
        "computed_from": {"overall_bias_pp": overall_bias, "n": int(len(eval_df))},
    }

    score_df = eval_df.groupby(["Ticker", "Region"]).agg(
        n=("hit", "size"),
        hit=("hit", "mean"),
        mae=("abs_error_pp", "mean"),
    ).reset_index()
    score_df["reliability"] = (0.6 * score_df["hit"].fillna(0.0) + 0.4 * (1.0 / (1.0 + score_df["mae"].fillna(5.0)))).clip(0, 1)
    score_df.to_csv(settings.SCORES_CSV, index=False)
    score_df.to_csv(settings.SCORES_CSV_REPORTS, index=False)

    mentor_status = "SKIPPED_NO_KEY"
    mentor_overrides = {"status": mentor_status, "items": [], "safe_overrides": {"clip_pct": 8.0, "multiplier_cap": 1.5}}

    if settings.OPENAI_API_KEY:
        mentor_overrides = run_mentor(eval_df, week_end=pd.Timestamp(last_target))
        mentor_status = mentor_overrides.get("status", "UNKNOWN")

    _write_json(settings.CALIBRATION_JSON, calib)
    _write_json(settings.CALIBRATION_JSON_REPORTS, calib)
    _write_json(settings.MENTOR_OVERRIDES_JSON, _jsonable(mentor_overrides))
    _write_json(settings.MENTOR_OVERRIDES_JSON_REPORTS, _jsonable(mentor_overrides))

    msg = [
        "StockD learning update",
        f"Week end: {pd.Timestamp(last_target).date()}",
        f"Eval rows: {len(eval_df)}",
        f"Mentor: {mentor_status}",
    ]
    for region, m in summ.items():
        msg.append(f"{region}: n={int(m['n'])}, hit={m['hit']:.2f}, MAE={m['MAE_pp']:.2f}pp, bias={m['bias_pp']:.2f}pp")

    send_telegram_message("\n".join(msg))
    send_telegram_document(settings.MODEL_EVAL_SUMMARY, caption="Model eval summary")
    send_telegram_document(settings.SCORES_CSV, caption="Ticker reliability scores")
    send_telegram_document(settings.CALIBRATION_JSON, caption="Calibration")
    send_telegram_document(settings.MENTOR_OVERRIDES_JSON, caption="Mentor overrides")


if __name__ == "__main__":
    run_learning()
