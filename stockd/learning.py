# stockd/learning.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from stockd import settings
from stockd.telegram_utils import send_telegram_message, send_telegram_document


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_date(x: Any) -> Optional[date]:
    try:
        if isinstance(x, pd.Timestamp):
            return x.date()
        if isinstance(x, str) and x:
            return pd.to_datetime(x).date()
    except Exception:
        return None
    return None


def _fmt(x: Any, d: int = 2) -> str:
    try:
        if x is None:
            return "-"
        if isinstance(x, float) and pd.isna(x):
            return "-"
        return f"{float(x):.{d}f}"
    except Exception:
        return "-"


def _build_region_summary_from_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizează summary pe regiuni, calculat direct din eval_df:
    Region, n, hit, MAE_pp, bias_pp
    """
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Region", "n", "hit", "MAE_pp", "bias_pp"])

    df = eval_df.copy()

    # numeric normalize
    for c in ["Model_ER_Pct", "Realized_Pct", "Error_Pct", "AbsError_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure AbsError_Pct
    if "AbsError_Pct" not in df.columns:
        if "Error_Pct" in df.columns:
            df["AbsError_Pct"] = df["Error_Pct"].abs()
        else:
            df["AbsError_Pct"] = pd.NA

    # ensure DirectionHit
    if "DirectionHit" not in df.columns and "Model_ER_Pct" in df.columns and "Realized_Pct" in df.columns:
        def sgn(v: float) -> int:
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0

        df["DirectionHit"] = (
            df["Model_ER_Pct"].fillna(0).apply(sgn) == df["Realized_Pct"].fillna(0).apply(sgn)
        ).astype(int)

    if "Region" not in df.columns:
        return pd.DataFrame(columns=["Region", "n", "hit", "MAE_pp", "bias_pp"])

    # groupby Region
    out = (
        df.groupby("Region", as_index=False)
        .agg(
            n=("AbsError_Pct", "count"),
            hit=("DirectionHit", "mean") if "DirectionHit" in df.columns else ("AbsError_Pct", "count"),
            MAE_pp=("AbsError_Pct", "mean"),
            bias_pp=("Error_Pct", "mean") if "Error_Pct" in df.columns else ("AbsError_Pct", "mean"),
        )
        .sort_values("Region")
        .reset_index(drop=True)
    )

    # enforce types
    out["n"] = out["n"].astype(int, errors="ignore")
    return out


def _compute_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scor simplu de reliability pe ticker:
    reliability = hit_rate * 1/(1+MAE/2)
    """
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    df = eval_df.copy()
    for c in ["Error_Pct", "AbsError_Pct", "Model_ER_Pct", "Realized_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "AbsError_Pct" not in df.columns and "Error_Pct" in df.columns:
        df["AbsError_Pct"] = df["Error_Pct"].abs()

    if "DirectionHit" not in df.columns and "Model_ER_Pct" in df.columns and "Realized_Pct" in df.columns:
        def sgn(v: float) -> int:
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0
        df["DirectionHit"] = (df["Model_ER_Pct"].fillna(0).apply(sgn) == df["Realized_Pct"].fillna(0).apply(sgn)).astype(int)

    if not {"Ticker", "Region"}.issubset(df.columns):
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    agg = (
        df.groupby(["Ticker", "Region"], as_index=False)
        .agg(
            n=("AbsError_Pct", "count"),
            hit_rate=("DirectionHit", "mean") if "DirectionHit" in df.columns else ("AbsError_Pct", "count"),
            mae_pp=("AbsError_Pct", "mean"),
            bias_pp=("Error_Pct", "mean") if "Error_Pct" in df.columns else ("AbsError_Pct", "mean"),
        )
    )

    def rel(row):
        hit = float(row.get("hit_rate", 0.0))
        mae = float(row.get("mae_pp", 0.0))
        shrink = 1.0 / (1.0 + max(0.0, mae) / 2.0)
        return max(0.0, min(1.0, hit * shrink))

    agg["reliability"] = agg.apply(rel, axis=1)
    return agg.sort_values(["Region", "reliability"], ascending=[True, False]).reset_index(drop=True)


def _compute_region_calibration(region_summary: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"regions": {}}
    if region_summary is None or region_summary.empty:
        return out

    for _, r in region_summary.iterrows():
        reg = str(r.get("Region", "")).strip()
        if not reg:
            continue
        n = int(r.get("n", 0) or 0)
        mae = float(r.get("MAE_pp", 0.0) or 0.0)
        bias = float(r.get("bias_pp", 0.0) or 0.0)

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

    prices = load_prices()
    forecasts = load_forecasts()
    eval_df = evaluate_weekly(prices, forecasts)

    # infer week_end from TargetDate if present
    week_end: date = date.today()
    if eval_df is not None and not eval_df.empty and "TargetDate" in eval_df.columns:
        mx = pd.to_datetime(eval_df["TargetDate"], errors="coerce").max()
        d = _safe_date(mx)
        if d:
            week_end = d

    # build region summary + scores
    region_summary = _build_region_summary_from_eval(eval_df)
    scores_df = _compute_scores(eval_df)

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = settings.REPORTS_DIR / "model_eval_detailed.csv"
    summary_path = settings.REPORTS_DIR / "model_eval_summary.csv"

    scores_path = getattr(settings, "SCORES_FILE", settings.DATA_DIR / "scores_stockd.csv")
    calibration_path = getattr(settings, "CALIBRATION_FILE", settings.DATA_DIR / "calibration.json")

    eval_df.to_csv(detailed_path, index=False)
    region_summary.to_csv(summary_path, index=False)

    _ensure_parent(Path(scores_path))
    scores_df.to_csv(scores_path, index=False)

    calibration = {
        "generated_at": date.today().isoformat(),
        "week_end": week_end.isoformat(),
        "region_calibration": _compute_region_calibration(region_summary),
    }
    _write_json(Path(calibration_path), calibration)

    # mentor (safe)
    mentor_status = "SKIPPED"
    mentor_err = ""
    mentor_md_path = None
    overrides_path = getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")

    try:
        from stockd.mentor import run_mentor
        mentor_info = run_mentor(eval_df, week_end=week_end)
        mentor_status = str(mentor_info.get("status", "OK"))
        mentor_err = str(mentor_info.get("error", ""))[:200]
        mentor_md_path = mentor_info.get("md_path")
    except Exception as e:
        mentor_status = "ERROR"
        mentor_err = f"{type(e).__name__}: {str(e)[:200]}"

    # Telegram message
    lines = []
    lines.append("StockD learning update")
    lines.append(f"Week end: {week_end.isoformat()}")
    lines.append(f"Eval rows: {0 if eval_df is None else len(eval_df)}")
    lines.append(f"Mentor: {mentor_status}" + (f" ({mentor_err})" if mentor_err else ""))

    for _, r in region_summary.iterrows():
        reg = str(r["Region"])
        n = int(r["n"]) if pd.notna(r["n"]) else 0
        hit = r["hit"]
        mae = r["MAE_pp"]
        bias = r["bias_pp"]
        lines.append(
            f"{reg}: n={n}, hit={_fmt(hit, 2)}, MAE={_fmt(mae, 2)}pp, bias={_fmt(bias, 2)}pp"
        )

    send_telegram_message("\n".join(lines))

    send_telegram_document(str(summary_path), caption="Model eval summary")
    send_telegram_document(str(scores_path), caption="Ticker reliability scores")
    send_telegram_document(str(calibration_path), caption="Calibration")

    if Path(overrides_path).exists():
        send_telegram_document(str(overrides_path), caption="Mentor overrides")

    if mentor_md_path:
        send_telegram_document(str(mentor_md_path), caption="Mentor postmortem")


if __name__ == "__main__":
    run_learning()
