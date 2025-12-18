# stockd/learning.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from stockd import settings
from stockd.telegram_utils import send_telegram_message, send_telegram_document


# ---------------------------
# Helpers
# ---------------------------

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
    """
    Fallback dacă evaluation.py nu are dedup_forecasts.
    Păstrează ultima predicție per (WeekStart, TargetDate, Ticker, Region).
    """
    if forecasts is None or forecasts.empty:
        return forecasts

    df = forecasts.copy()
    # normalize cols
    for c in ["Date", "WeekStart", "TargetDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    keys = [c for c in ["WeekStart", "TargetDate", "Ticker", "Region"] if c in df.columns]
    if "Date" in df.columns:
        df = df.sort_values("Date")
    if keys:
        df = df.groupby(keys, as_index=False).tail(1)
    return df


def _compute_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construiește un scor simplu de fiabilitate per (Ticker, Region).
    Output: data/scores_stockd.csv
    """
    if eval_df is None or eval_df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    df = eval_df.copy()
    for c in ["Error_Pct", "AbsError_Pct", "Model_ER_Pct", "Realized_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # DirectionHit poate lipsi, îl calculăm dacă avem Model/Real
    if "DirectionHit" not in df.columns and "Model_ER_Pct" in df.columns and "Realized_Pct" in df.columns:
        df["DirectionHit"] = (df["Model_ER_Pct"].fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) ==
                              df["Realized_Pct"].fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).astype(int)

    gcols = [c for c in ["Ticker", "Region"] if c in df.columns]
    if len(gcols) < 2:
        return pd.DataFrame(columns=["Ticker", "Region", "n", "hit_rate", "mae_pp", "bias_pp", "reliability"])

    agg = df.groupby(gcols).agg(
        n=("AbsError_Pct", "count"),
        hit_rate=("DirectionHit", "mean") if "DirectionHit" in df.columns else ("AbsError_Pct", "count"),
        mae_pp=("AbsError_Pct", "mean") if "AbsError_Pct" in df.columns else ("Error_Pct", "mean"),
        bias_pp=("Error_Pct", "mean") if "Error_Pct" in df.columns else ("AbsError_Pct", "mean"),
    ).reset_index()

    # reliability: combinație simplă între hit și mae
    # mai mare MAE => penalizare mai puternică
    def rel(row):
        hit = float(row.get("hit_rate", 0.0))
        mae = float(row.get("mae_pp", 0.0))
        shrink = 1.0 / (1.0 + max(0.0, mae) / 2.0)  # 2pp ca “scale” inițial
        return max(0.0, min(1.0, hit * shrink))

    agg["reliability"] = agg.apply(rel, axis=1)
    return agg.sort_values(["Region", "reliability"], ascending=[True, False])


def _compute_region_calibration(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construiește calibrare pe regiune: bias + shrink.
    Acceptă diverse nume de coloane.
    """
    out: Dict[str, Any] = {"regions": {}}
    if summary_df is None or summary_df.empty:
        return out

    df = summary_df.copy()
    df.columns = [str(c) for c in df.columns]

    if "Region" not in df.columns:
        return out

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    n_col = pick(["n", "N", "count"])
    mae_col = pick(["MAE_pp", "MAE", "mae_pp"])
    bias_col = pick(["bias_pp", "Bias_pp", "bias"])

    for _, r in df.iterrows():
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


# ---------------------------
# Main
# ---------------------------

def run_learning() -> None:
    # Importă ce există în evaluation.py (nume reale din repo)
    from stockd.evaluation import load_prices, load_forecasts, evaluate_weekly
    try:
        from stockd.evaluation import dedup_forecasts  # optional
    except Exception:
        dedup_forecasts = None  # type: ignore

    try:
        from stockd.evaluation import summarize  # optional
    except Exception:
        summarize = None  # type: ignore

    # Load
    prices = load_prices()
    forecasts = load_forecasts()

    # Dedup
    if dedup_forecasts is not None:
        forecasts = dedup_forecasts(forecasts)
    else:
        forecasts = _dedup_forecasts_local(forecasts)

    # Evaluate
    eval_df = evaluate_weekly(prices, forecasts)

    # Derive week_end
    week_end: date = date.today()
    if eval_df is not None and not eval_df.empty:
        if "TargetDate" in eval_df.columns:
            mx = pd.to_datetime(eval_df["TargetDate"], errors="coerce").max()
            d = _safe_date(mx)
            if d:
                week_end = d

    # Summary
    if summarize is not None:
        summary_df = summarize(eval_df)
        # unele implementări întorc dict; normalizăm
        if isinstance(summary_df, dict):
            summary_df = pd.DataFrame(summary_df)
    else:
        # fallback minimal: sumar pe regiune
        summary_df = pd.DataFrame()
        if eval_df is not None and not eval_df.empty and "Region" in eval_df.columns:
            tmp = eval_df.copy()
            for c in ["AbsError_Pct", "Error_Pct", "DirectionHit"]:
                if c in tmp.columns:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            if "DirectionHit" not in tmp.columns and "Model_ER_Pct" in tmp.columns and "Realized_Pct" in tmp.columns:
                tmp["DirectionHit"] = (
                    tmp["Model_ER_Pct"].fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) ==
                    tmp["Realized_Pct"].fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                ).astype(int)

            summary_df = tmp.groupby("Region").agg(
                n=("AbsError_Pct", "count") if "AbsError_Pct" in tmp.columns else ("Error_Pct", "count"),
                hit=("DirectionHit", "mean") if "DirectionHit" in tmp.columns else ("Error_Pct", "count"),
                MAE_pp=("AbsError_Pct", "mean") if "AbsError_Pct" in tmp.columns else ("Error_Pct", "mean"),
                bias_pp=("Error_Pct", "mean") if "Error_Pct" in tmp.columns else ("AbsError_Pct", "mean"),
            ).reset_index()

    # Scores
    scores_df = _compute_scores(eval_df)

    # Paths
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detailed_path = settings.REPORTS_DIR / "model_eval_detailed.csv"
    summary_path = settings.REPORTS_DIR / "model_eval_summary.csv"
    scores_path = getattr(settings, "SCORES_FILE", settings.DATA_DIR / "scores_stockd.csv")

    # Save artifacts
    eval_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _ensure_parent(scores_path)
    scores_df.to_csv(scores_path, index=False)

    # Calibration
    calibration_path = getattr(settings, "CALIBRATION_FILE", settings.DATA_DIR / "calibration.json")
    calibration = {
        "generated_at": date.today().isoformat(),
        "week_end": week_end.isoformat(),
        "region_calibration": _compute_region_calibration(summary_df),
    }
    _write_json(Path(calibration_path), calibration)

    # Mentor (optional)
    mentor_status = "SKIPPED"
    mentor_md_path = None
    try:
        from stockd.mentor import run_mentor
        mentor_info = run_mentor(eval_df, week_end=week_end)
        mentor_status = str(mentor_info.get("status", "OK"))
        mentor_md_path = mentor_info.get("md_path")
    except Exception as e:
        mentor_status = f"ERROR: {type(e).__name__}"

    # Telegram notify
    msg_lines = []
    msg_lines.append("StockD learning update")
    msg_lines.append(f"Week end: {week_end.isoformat()}")
    msg_lines.append(f"Eval rows: {0 if eval_df is None else len(eval_df)}")
    msg_lines.append(f"Mentor: {mentor_status}")

    # Add region summary compact
    if summary_df is not None and not summary_df.empty:
        try:
            for _, r in summary_df.iterrows():
                reg = str(r.get("Region", ""))
                n = r.get("n", r.get("N", ""))
                hit = r.get("hit", r.get("Hit", ""))
                mae = r.get("MAE_pp", r.get("MAE", ""))
                bias = r.get("bias_pp", r.get("bias", ""))
                msg_lines.append(f"{reg}: n={n}, hit={hit}, MAE={mae}pp, bias={bias}pp")
        except Exception:
            pass

    send_telegram_message("\n".join(msg_lines))

    # Send artifacts
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
