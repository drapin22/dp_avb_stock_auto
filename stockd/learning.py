from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from stockd import settings
from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly, dedup_forecasts, summarize
from stockd.features import compute_ticker_features
from stockd.macro import get_macro_snapshot
from stockd.online_model import save_state
from stockd.calibration import build_region_calibration, save_calibration
from stockd.scoring import compute_scores, save_scores
from stockd.mentor import postmortem_and_rules
from stockd.telegram_utils import send_chunked_message, send_telegram_document


FEATURE_COLS = [
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "max_dd_60d",
    "beta",
    "vix_level",
    "dxy_ret_5d",
    "oil_ret_5d",
    "gold_ret_5d",
    "bench_ret_5d",
]


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Fit ridge regression with intercept using closed-form solution.
    Returns (coef, intercept).
    """
    if X.size == 0 or y.size == 0:
        return np.zeros((X.shape[1],), dtype=float), 0.0

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # add intercept column
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xa = np.concatenate([ones, X], axis=1)

    # ridge (do not penalize intercept)
    I = np.eye(Xa.shape[1], dtype=float)
    I[0, 0] = 0.0

    A = Xa.T @ Xa + l2 * I
    b = Xa.T @ y

    w = np.linalg.solve(A, b)
    intercept = float(w[0])
    coef = w[1:].astype(float)
    return coef, intercept


def _append_csv(path: Path, df_new: pd.DataFrame, dedup_keys: List[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, df_new], ignore_index=True)
    else:
        combined = df_new.copy()

    # normalize datetime columns if present
    for c in ["WeekStart", "TargetDate"]:
        if c in combined.columns:
            combined[c] = pd.to_datetime(combined[c], errors="coerce")

    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    combined.to_csv(path, index=False)
    return combined


def _latest_completed_target(forecasts: pd.DataFrame) -> pd.Timestamp | None:
    if forecasts is None or forecasts.empty:
        return None
    f = forecasts.copy()
    f["TargetDate"] = pd.to_datetime(f["TargetDate"], errors="coerce")
    today = pd.Timestamp(date.today())
    completed = f[f["TargetDate"] < today]
    if completed.empty:
        return None
    return completed["TargetDate"].max()


def run_learning() -> None:
    prices = load_prices()
    forecasts = load_forecasts()
    forecasts = dedup_forecasts(forecasts)

    if prices.empty or forecasts.empty:
        send_chunked_message("StockD learning: missing prices or forecasts. Skipping.")
        return

    last_target = _latest_completed_target(forecasts)
    if last_target is None:
        send_chunked_message("StockD learning: no completed forecast window yet. Skipping.")
        return

    f_week = forecasts[forecasts["TargetDate"] == last_target].copy()
    eval_df = evaluate_weekly(prices, f_week)
    if eval_df.empty:
        send_chunked_message(f"StockD learning: evaluation empty for TargetDate={last_target.date()}. Skipping.")
        return

    ws = pd.to_datetime(eval_df["WeekStart"].iloc[0]).date()
    we = pd.to_datetime(eval_df["TargetDate"].iloc[0]).date()

    # Append evaluation history (reports/)
    eval_hist = _append_csv(
        settings.MODEL_EVAL_DETAILED,
        eval_df,
        dedup_keys=["WeekStart", "TargetDate", "Ticker", "Region", "ModelVersion"],
    )

    summary_df = summarize(eval_df)
    summary_df.to_csv(settings.MODEL_EVAL_SUMMARY, index=False)

    # Train/update online model on this week's realized returns
    feats = compute_ticker_features(prices, as_of=pd.Timestamp(we))
    train = eval_df.merge(feats, on=["Ticker", "Region"], how="left")

    macro = get_macro_snapshot(as_of=date.today())
    for k in ["vix_level", "dxy_ret_5d", "oil_ret_5d", "gold_ret_5d", "bench_ret_5d"]:
        train[k] = float(macro.get(k, 0.0) or 0.0)

    # Fill missing numeric
    for c in FEATURE_COLS:
        train[c] = pd.to_numeric(train.get(c), errors="coerce").fillna(0.0)

    y = pd.to_numeric(train["Realized_Pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X = train[FEATURE_COLS].to_numpy(dtype=float)

    coef, intercept = _ridge_fit(X, y, l2=1.0)
    state = {
        "feature_cols": FEATURE_COLS,
        "coef": coef.tolist(),
        "intercept": intercept,
        "trained_on_target_date": str(we),
        "n_samples": int(X.shape[0]),
    }
    save_state(state)

    # Calibration + scoring (data/)
    calib = build_region_calibration(eval_hist)
    save_calibration(calib)

    scores = compute_scores(eval_hist)
    save_scores(scores)

    # Optional mentor: produce postmortem + rules
    mentor_md_path = settings.REPORTS_DIR / f"mentor_postmortem_{we}.md"
    rules_json_path = settings.MENTOR_OVERRIDES_JSON

    if settings.ENABLE_LLM_POSTMORTEM and settings.OPENAI_API_KEY:
        try:
            worst = eval_df.sort_values("AbsError_Pct", ascending=False).head(12)
            payload = worst.to_dict(orient="records")
            for r in payload:
                for k, v in list(r.items()):
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        r[k] = str(pd.to_datetime(v).date())
            post = postmortem_and_rules(payload, macro=macro, as_of=date.today())

            mentor_md_path.write_text(post.get("markdown", ""), encoding="utf-8")
            rules_json_path.write_text(json.dumps(post.get("rules", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # Telegram outputs
    msg = (
        "StockD learning update\n"
        f"Window: {ws} → {we}\n"
        f"Tickers: {eval_df.shape[0]}\n"
        f"MAE: {float(eval_df['AbsError_Pct'].mean()):.2f}%\n"
        f"Hit rate: {float(eval_df['DirectionHit'].mean())*100:.1f}%\n"
        "Artifacts: model_state.json, calibration.json, scores_stockd.csv, model_eval_detailed.csv\n"
    )
    send_chunked_message(msg)

    # Attach key artifacts (best-effort)
    for p in [
        settings.MODEL_EVAL_DETAILED,
        settings.MODEL_EVAL_SUMMARY,
        settings.CALIBRATION_JSON,
        settings.SCORES_CSV,
        settings.MODEL_STATE_JSON,
        mentor_md_path,
        rules_json_path,
    ]:
        if Path(p).exists():
            send_telegram_document(p, caption=Path(p).name)


if __name__ == "__main__":
    run_learning()
