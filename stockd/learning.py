# stockd/learning.py
from __future__ import annotations

import json
from datetime import date
from typing import Dict, Any

import pandas as pd

from stockd import settings
from stockd.telegram_utils import (
    send_telegram_message,
    send_telegram_document,
)
from stockd.evaluation import (
    load_prices_history,
    load_forecasts,
    dedup_forecasts,
    evaluate_weekly,
    summarize_eval,
    compute_ticker_scores,
)
from stockd.mentor import run_mentor


def _json_dump(path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _today_iso() -> str:
    return date.today().isoformat()


def _compute_region_calibration(summary_df: pd.DataFrame) -> Dict[str, Any]:
    # summary_df: columns like Region, n, MAE_pp, bias_pp (în funcție de cum ai implementat)
    # Facem reguli simple, robuste:
    # region_bias = bias_pp
    # region_shrink = clamp( 1 / (1 + MAE_pp/2) , 0.25, 1.0 )
    out: Dict[str, Any] = {"regions": {}}

    if summary_df is None or summary_df.empty:
        return out

    df = summary_df.copy()
    df.columns = [str(c) for c in df.columns]

    # acceptăm mai multe naming-uri posibile
    region_col = "Region" if "Region" in df.columns else None
    if region_col is None:
        return out

    def pick(col_candidates, default=0.0):
        for c in col_candidates:
            if c in df.columns:
                return c
        return None

    n_col = pick(["n", "N", "count"])
    mae_col = pick(["MAE_pp", "MAE", "mae_pp"])
    bias_col = pick(["bias_pp", "Bias_pp", "bias"])

    for _, r in df.iterrows():
        region = str(r.get(region_col, "")).strip()
        if not region:
            continue

        n = int(r.get(n_col, 0)) if n_col else 0
        mae = float(r.get(mae_col, 0.0)) if mae_col else 0.0
        bias = float(r.get(bias_col, 0.0)) if bias_col else 0.0

        # shrink: mai mare MAE, mai mult shrink
        shrink = 1.0 / (1.0 + (mae / 2.0 if mae > 0 else 0.0))
        shrink = max(0.25, min(1.0, shrink))

        out["regions"][region] = {
            "n": n,
            "mae_pp": round(mae, 4),
            "bias_pp": round(bias, 4),
            "shrink": round(shrink, 4),
        }

    return out


def run_learning() -> None:
    prices = load_prices_history(settings.PRICES_FILE)
    forecasts = load_forecasts(settings.FORECASTS_FILE)
    forecasts = dedup_forecasts(forecasts)

    eval_df, info = evaluate_weekly(prices, forecasts)
    summary_df = summarize_eval(eval_df)
    scores_df = compute_ticker_scores(eval_df)

    # Persist artifacts
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = settings.REPORTS_DIR / "model_eval_detailed.csv"
    sum_path = settings.REPORTS_DIR / "model_eval_summary.csv"
    scores_path = settings.SCORES_FILE

    eval_df.to_csv(eval_path, index=False)
    summary_df.to_csv(sum_path, index=False)
    scores_df.to_csv(scores_path, index=False)

    # Calibration.json written in repo/data
    calibration = {
        "generated_at": _today_iso(),
        "region_calibration": _compute_region_calibration(summary_df),
    }
    _json_dump(settings.CALIBRATION_FILE, calibration)

    # Mentor
    week_end = info["week_end"]
    mentor_info = run_mentor(eval_df, week_end=week_end)

    # Telegram notify
    lines = []
    lines.append("StockD learning update")
    lines.append(f"Eval rows: {len(eval_df)}")
    # tipic: summary_df are Region, hit, MAE, bias
    # păstrăm format tolerant
    if summary_df is not None and not summary_df.empty:
        for _, r in summary_df.iterrows():
            reg = str(r.get("Region", ""))
            n = r.get("n", r.get("N", ""))
            hit = r.get("hit", r.get("Hit", ""))
            mae = r.get("MAE_pp", r.get("MAE", ""))
            bias = r.get("bias_pp", r.get("bias", ""))
            lines.append(f"{reg}: n={n}, hit={hit}, MAE={mae}pp, bias={bias}pp")
    lines.append(f"Mentor: {mentor_info.get('status')}")

    send_telegram_message("\n".join(lines))

    # Send artifacts
    send_telegram_document(str(sum_path), caption="Model eval summary")
    send_telegram_document(str(scores_path), caption="Ticker reliability scores")
    send_telegram_document(str(settings.CALIBRATION_FILE), caption="Calibration")

    overrides_path = settings.MENTOR_OVERRIDES_FILE
    if overrides_path.exists():
        send_telegram_document(str(overrides_path), caption="Mentor overrides")

    md_path = mentor_info.get("md_path")
    if md_path:
        send_telegram_document(md_path, caption="Mentor postmortem")


if __name__ == "__main__":
    run_learning()
