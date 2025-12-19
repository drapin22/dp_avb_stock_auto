from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from stockd import settings
from stockd.telegram_utils import send_chunked_message, send_telegram_document, send_telegram_message


def _next_monday(d: datetime) -> datetime:
    days = (7 - d.weekday()) % 7
    if days == 0:
        days = 7
    return d + timedelta(days=days)


def _week_dates(anchor: datetime | None = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    now = anchor or datetime.utcnow()
    week_start = _next_monday(now)
    week_end = week_start + timedelta(days=4)
    return pd.Timestamp(week_start.date()), pd.Timestamp(week_end.date())


def _load_universe() -> pd.DataFrame:
    rows = []
    for region, p in [("RO", settings.HOLDINGS_RO), ("EU", settings.HOLDINGS_EU), ("US", settings.HOLDINGS_US)]:
        if p.exists():
            df = pd.read_csv(p)
            if "Ticker" in df.columns:
                for t in df["Ticker"].dropna().astype(str).tolist():
                    rows.append({"Ticker": t.strip(), "Region": region})
    return pd.DataFrame(rows).drop_duplicates()


def _read_json(path: Path) -> Dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _read_scores_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame(columns=["Ticker", "Region", "reliability"])


def _apply_calibration_scoring_vol(raw: pd.DataFrame, calib: Dict, scores: pd.DataFrame, mentor_safe: Dict) -> pd.DataFrame:
    out = raw.copy()

    if scores is None or scores.empty or "reliability" not in scores.columns:
        out["reliability"] = 0.5
    else:
        sc = scores.copy()
        sc["Ticker"] = sc["Ticker"].astype(str)
        sc["Region"] = sc["Region"].astype(str)
        sc["reliability"] = pd.to_numeric(sc["reliability"], errors="coerce")
        sc = sc.dropna(subset=["reliability"])
        out = out.merge(sc[["Ticker", "Region", "reliability"]], on=["Ticker", "Region"], how="left")
        out["reliability"] = pd.to_numeric(out["reliability"], errors="coerce").fillna(0.5)

    mult = float(calib.get("global_multiplier", 1.0) or 1.0)
    clip = float(calib.get("clip_pct", 8.0) or 8.0)

    safe_mult_cap = float(mentor_safe.get("multiplier_cap", 1.5) or 1.5)
    safe_clip_pct = float(mentor_safe.get("clip_pct", clip) or clip)

    mult = min(mult, safe_mult_cap)
    clip = min(clip, safe_clip_pct)

    gate = 0.5 + 0.5 * out["reliability"].clip(0, 1)
    out["ER_Pct_Adj"] = (out["ER_Pct"] * mult * gate).clip(-clip, clip)

    return out


def run_weekly_forecast() -> None:
    week_start, week_end = _week_dates()

    uni = _load_universe()
    if uni.empty:
        send_telegram_message("StockD weekly forecast: universe is empty (no holdings files).")
        return

    raw = uni.copy()
    raw["Date"] = pd.Timestamp(datetime.utcnow().date())
    raw["WeekStart"] = week_start
    raw["TargetDate"] = week_end
    raw["ModelVersion"] = "StockD_WEEKLY_BASE"
    raw["HorizonDays"] = 5
    raw["ER_Pct"] = 0.0
    raw["Notes"] = "baseline neutral (will be calibrated and scored)"

    calib = _read_json(settings.CALIBRATION_JSON)
    scores = _read_scores_csv(settings.SCORES_CSV)
    mentor_overrides = _read_json(settings.MENTOR_OVERRIDES_JSON)
    mentor_safe = (mentor_overrides.get("safe_overrides") or {}) if isinstance(mentor_overrides, dict) else {}

    adj = _apply_calibration_scoring_vol(raw, calib, scores, mentor_safe)

    out_cols = ["Date", "WeekStart", "TargetDate", "ModelVersion", "Ticker", "Region", "HorizonDays", "ER_Pct", "Notes"]
    to_save = adj.copy()
    to_save["ER_Pct"] = to_save["ER_Pct_Adj"]
    to_save["Notes"] = "weekly forecast (adj=calib*reliability, clipped)"
    to_save = to_save[out_cols]

    if settings.FORECASTS_STOCKD.exists():
        old = pd.read_csv(settings.FORECASTS_STOCKD)
        df = pd.concat([old, to_save], ignore_index=True)
    else:
        df = to_save

    settings.FORECASTS_STOCKD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.FORECASTS_STOCKD, index=False)

    weekly_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start.date()}_{week_end.date()}.csv"
    to_save.to_csv(weekly_csv, index=False)

    header = f"StockD weekly forecast\nWeek: {week_start.date()} → {week_end.date()}\nUniverse: {len(adj)} tickers"
    lines = [header, "", "Signals (Adj):"]
    show_df = adj.sort_values("ER_Pct_Adj", ascending=False)
    for _, r in show_df.iterrows():
        lines.append(f"{r['Ticker']} ({r['Region']}): {r['ER_Pct_Adj']:+.2f}% (rel={r['reliability']:.2f})")

    if settings.TELEGRAM_SEND_ALL_SIGNALS:
        send_chunked_message("\n".join(lines))
    else:
        send_telegram_message("\n".join(lines[:20]))
        send_telegram_message("Full forecast is attached as CSV.")

    send_telegram_document(weekly_csv, caption=f"Full forecast list: {week_start.date()} → {week_end.date()}")


if __name__ == "__main__":
    run_weekly_forecast()
