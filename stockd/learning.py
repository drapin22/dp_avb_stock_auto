# stockd/learning.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from stockd import settings
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def _safe_date(x: Any) -> Optional[date]:
    try:
        if isinstance(x, pd.Timestamp):
            return x.date()
        if isinstance(x, str) and x:
            return pd.to_datetime(x).date()
    except Exception:
        return None
    return None


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dedup_forecasts_local(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts is None or forecasts.empty:
        return forecasts

    df = forecasts.copy()
    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    keys = [c for c in ["WeekStart", "TargetDate", "Ticker", "Region"] if c in df.columns]
    if "Date" in df.columns:
        df = df.sort_values("Date")
    if keys:
        df = df.groupby(keys, as_index=False).tail(1)
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _format_num(x: Any, decimals: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "-"


def _compute_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    df = eval_df.copy()
    for c in ["Error_Pct", "AbsError_Pct", "Model_ER_Pct", "Realized_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "DirectionHit" not in df.columns and "Model_ER_Pct" in df.columns and "Realized_Pct" in df.columns:
        def sgn(v: float) -> int:
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0
        df["DirectionHit"] = (df["Model_ER_Pct"].fillna(0).apply(sgn) == df["Realized_Pct"].fillna(0).apply(sgn)).astype(int)

    gcols = [c for c in ["Ticker", "Region"] if c in df.columns]
    if len(gcols) < 2:
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    mae_source = "AbsError_Pct" if "AbsError_Pct" in df.columns else "Error_Pct"
    bias_source = "Error_Pct" if "Error_Pct" in df.columns else mae_source

    agg = df.groupby(gcols).agg(
        n=(mae_source, "count"),
        hit_rate=("DirectionHit", "mean") if "DirectionHit" in df.columns else (mae_source, "count"),
        mae_pp=(mae_source, "mean"),
        bias_pp=(bias_source, "mean"),
    ).reset_index()

    def rel(row):
        hit = float(row.get("hit_rate", 0.0))
        mae = float(row.get("mae_pp", 0.0))
        shrink = 1.0 / (1.0 + max(0.0, mae) / 2.0)
        return max(0.0, min(1.0, hit * shrink))

    agg["reliability"] = agg.apply(rel, axis=1)
    return agg.sort_values(["Region", "reliability"], ascending=[True, False])


def _compute_region_calibration(summary_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"regions": {}}
    if summary_df is None or summary_df.empty or "Region" not in summary_df.columns:
        return out

    n_col = _pick_col(summary_df, ["n", "N", "count"])
    hit_col = _pick_col(summary_df, ["hit", "hit_rate", "HitRate", "hit_rate_raw", "hit_rate_calibrated"])
    mae_col = _pick_col(summary_df, ["MAE_pp", "mae_pp", "MAE", "mae", "MAE_raw", "MAE_calibrated"])
    bias_col = _pick_col(summary_df, ["bias_pp", "Bias_pp", "bias", "Bias"])

    for _, r in summary_df.iterrows():
        reg = str(r.get("Region", "")).strip()
        if not reg:
            continue

        n = int(r.get(n_col, 0)) if n_col else 0
        mae = float(r.get(mae_col, 0.0)) if mae_col else 0.0
        bias = float(r.get(bias_col, 0.0)) if bias_col else 0.0

        shrink = 1.0 / (1.0 + (mae / 2.0 if mae > 0 else 0.0))
        shrink = max(0.25, min(1.0, shrink))

        out["regions"][reg] = {
            "n": n,
            "mae_pp": round(mae, 6),
            "bias_pp": round(bias, 6),
            "shrink": round(shrink, 6),
        }

    return out


def run_learning() -> None:
    from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly
    try:
        from stockd.evaluation import dedup_forecasts  # optional
    except Exception:
        dedup_forecasts = None

    try:
        from stockd.evaluation import summarize  # optional
    except Exception:
        summarize = None

    prices = load_prices()
    forecasts = load_forecasts()

    if dedup_forecasts is not None:
        forecasts = dedup_forecasts(forecasts)
    else:
        forecasts = _dedup_forecasts_local(forecasts)

    eval_df = evaluate_weekly(prices, forecasts)

    week_end: date = date.today()
    if eval_df is not None and not eval_df.empty and "TargetDate" in eval_df.columns:
        mx = pd.to_datetime(eval_df["TargetDate"], errors="coerce").max()
        d = _safe_date(mx)
        if d:
            week_end = d

    if summarize is not None:
        summary_df = summarize(eval_df)
        if isinstance(summary_df, dict):
            summary_df = pd.DataFrame(summary_df)
    else:
        summary_df = pd.DataFrame()

    # De-dup summary by Region if duplicates exist
    if summary_df is not None and not summary_df.empty and "Region" in summary_df.columns:
        # keep last row per Region (or mean numeric if multiple)
        num_cols = [c for c in summary_df.columns if c != "Region" and pd.api.types.is_numeric_dtype(summary_df[c])]
        if num_cols:
            summary_df = summary_df.groupby("Region", as_index=False)[num_cols].mean()
        else:
            summary_df = summary_df.drop_duplicates(subset=["Region"], keep="last")

    scores_df = _compute_scores(eval_df)

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detailed_path = settings.REPORTS_DIR / "model_eval_detailed.csv"
    summary_path = settings.REPORTS_DIR / "model_eval_summary.csv"
    scores_path = getattr(settings, "SCORES_FILE", settings.DATA_DIR / "scores_stockd.csv")

    eval_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _ensure_parent(Path(scores_path))
    scores_df.to_csv(scores_path, index=False)

    calibration_path = getattr(settings, "CALIBRATION_FILE", settings.DATA_DIR / "calibration.json")
    calibration = {
        "generated_at": date.today().isoformat(),
        "week_end": week_end.isoformat(),
        "region_calibration": _compute_region_calibration(summary_df),
    }
    _write_json(Path(calibration_path), calibration)

    mentor_status = "SKIPPED"
    mentor_err = ""
    mentor_md_path = None
    try:
        from stockd.mentor import run_mentor
        mentor_info = run_mentor(eval_df, week_end=week_end)
        mentor_status = str(mentor_info.get("status", "OK"))
        mentor_md_path = mentor_info.get("md_path")
        mentor_err = str(mentor_info.get("error", ""))[:200]
    except Exception as e:
        mentor_status = "ERROR"
        mentor_err = f"{type(e).__name__}: {str(e)[:200]}"

    # Build message with robust column detection
    msg_lines = []
    msg_lines.append("StockD learning update")
    msg_lines.append(f"Week end: {week_end.isoformat()}")
    msg_lines.append(f"Eval rows: {0 if eval_df is None else len(eval_df)}")
    msg_lines.append(f"Mentor: {mentor_status}" + (f" ({mentor_err})" if mentor_err else ""))

    if summary_df is not None and not summary_df.empty and "Region" in summary_df.columns:
        n_col = _pick_col(summary_df, ["n", "N", "count"])
        hit_col = _pick_col(summary_df, ["hit", "hit_rate", "HitRate", "hit_rate_raw", "hit_rate_calibrated"])
        mae_col = _pick_col(summary_df, ["MAE_pp", "mae_pp", "MAE", "mae", "MAE_raw", "MAE_calibrated"])
        bias_col = _pick_col(summary_df, ["bias_pp", "Bias_pp", "bias", "Bias"])

        for _, r in summary_df.iterrows():
            reg = str(r.get("Region", "")).strip()
            if not reg:
                continue
            n = r.get(n_col) if n_col else None
            hit = r.get(hit_col) if hit_col else None
            mae = r.get(mae_col) if mae_col else None
            bias = r.get(bias_col) if bias_col else None

            msg_lines.append(
                f"{reg}: n={n if n is not None else '-'}, "
                f"hit={_format_num(hit, 2)}, "
                f"MAE={_format_num(mae, 2)}pp, "
                f"bias={_format_num(bias, 2)}pp"
            )

    send_telegram_message("\n".join(msg_lines))

    send_telegram_document(str(summary_path), caption="Model eval summary")
    send_telegram_document(str(scores_path), caption="Ticker reliability scores")
    send_telegram_document(str(calibration_path), caption="Calibration")

    overrides_path = getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")
    if Path(overrides_path).exists():
        send_telegram_document(str(overrides_path), caption="Mentor overrides")

    if mentor_md_path:
        send_telegram_document(str(mentor_md_path), caption="Mentor postmortem")


if __name__ == "__main__":
    run_learning()
