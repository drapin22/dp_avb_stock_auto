# stockd/model_weekly.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

from stockd import settings
from stockd.telegram_utils import send_telegram_message, send_telegram_document


@dataclass
class CalibRegion:
    shrink: float = 1.0
    bias_pp: float = 0.0


@dataclass
class MentorSafe:
    clip_pct: float = 8.0          # default clip
    multiplier_cap: float = 1.5    # default cap


def _week_bounds(today: Optional[date] = None) -> Tuple[date, date]:
    d = today or date.today()
    # week start Monday, target Friday
    ws = d - timedelta(days=d.weekday())
    we = ws + timedelta(days=4)
    return ws, we


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _load_calibration() -> Dict[str, CalibRegion]:
    p = getattr(settings, "CALIBRATION_FILE", settings.DATA_DIR / "calibration.json")
    raw = _load_json(Path(p))
    out: Dict[str, CalibRegion] = {}
    regions = raw.get("region_calibration", {}).get("regions", {})
    for reg, info in regions.items():
        try:
            out[str(reg)] = CalibRegion(
                shrink=float(info.get("shrink", 1.0)),
                bias_pp=float(info.get("bias_pp", 0.0)),
            )
        except Exception:
            out[str(reg)] = CalibRegion()
    return out


def _load_scores() -> pd.DataFrame:
    p = getattr(settings, "SCORES_FILE", settings.DATA_DIR / "scores_stockd.csv")
    if Path(p).exists():
        df = pd.read_csv(p)
        for c in ["reliability", "mae_pp", "bias_pp", "hit_rate"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    return pd.DataFrame(columns=["Ticker", "Region", "reliability", "mae_pp", "bias_pp", "hit_rate", "n"])


def _load_mentor_safe() -> MentorSafe:
    """
    mentor_overrides.json are struct:
    {
      "status": "...",
      "items": [...],
      "global_notes": "...",
      ...
    }
    În global_notes s-ar putea să fie JSON stringificat.
    Ne interesează safe_overrides: {clip_pct, multiplier_cap}
    """
    p = getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")
    raw = _load_json(Path(p))

    clip_pct = 8.0
    multiplier_cap = 1.5

    g = raw.get("global_notes")
    if isinstance(g, str) and g.strip():
        try:
            gobj = json.loads(g)
            so = gobj.get("safe_overrides", {})
            clip_pct = float(so.get("clip_pct", clip_pct))
            multiplier_cap = float(so.get("multiplier_cap", multiplier_cap))
        except Exception:
            pass

    return MentorSafe(clip_pct=clip_pct, multiplier_cap=multiplier_cap)


def _apply_calibration_and_scoring(
    df: pd.DataFrame,
    calib: Dict[str, CalibRegion],
    scores: pd.DataFrame,
    mentor_safe: MentorSafe,
) -> pd.DataFrame:
    """
    Input df must have: Ticker, Region, ER_Pct (raw)
    Output adds:
      BiasAdj, Shrink, Reliability, AdjERPct
    Rule:
      bias_corrected = ER_Pct - bias_pp(region)
      shrunk = bias_corrected * shrink(region)
      reliability gate:
        if reliability < 0.25 => multiply 0.5
        if reliability < 0.15 => force 0 (neutral)
      mentor caps:
        AdjERPct clipped to +/- clip_pct
        multiplier cap is applied vs raw magnitude: |Adj| <= |raw| * multiplier_cap + 0.25
    """
    out = df.copy()
    out["ER_Pct"] = pd.to_numeric(out["ER_Pct"], errors="coerce").fillna(0.0)

    # join scores
    if not scores.empty:
        out = out.merge(
            scores[["Ticker", "Region", "reliability"]],
            on=["Ticker", "Region"],
            how="left",
        )
    else:
        out["reliability"] = pd.NA

    out["reliability"] = pd.to_numeric(out["reliability"], errors="coerce").fillna(0.5)

    # region calibration
    def reg_shrink(r: str) -> float:
        return calib.get(r, CalibRegion()).shrink

    def reg_bias(r: str) -> float:
        return calib.get(r, CalibRegion()).bias_pp

    out["Shrink"] = out["Region"].astype(str).apply(reg_shrink)
    out["Bias_pp"] = out["Region"].astype(str).apply(reg_bias)

    out["BiasAdj"] = out["ER_Pct"] - out["Bias_pp"]
    out["AdjERPct"] = out["BiasAdj"] * out["Shrink"]

    # reliability gating
    def gate(row) -> float:
        rel = float(row["reliability"])
        v = float(row["AdjERPct"])
        if rel < 0.15:
            return 0.0
        if rel < 0.25:
            return v * 0.5
        return v

    out["AdjERPct"] = out.apply(gate, axis=1)

    # mentor safe caps
    clip = float(mentor_safe.clip_pct)
    cap = float(mentor_safe.multiplier_cap)

    def cap_fn(row) -> float:
        raw = float(row["ER_Pct"])
        adj = float(row["AdjERPct"])
        # clip absolute
        adj = max(-clip, min(clip, adj))
        # multiplier cap vs raw magnitude (with small floor so raw=0 doesn't kill everything)
        limit = abs(raw) * cap + 0.25
        adj = max(-limit, min(limit, adj))
        return adj

    out["AdjERPct"] = out.apply(cap_fn, axis=1)
    return out


def _load_universe() -> pd.DataFrame:
    # holdings files in data/
    dfs = []
    for fname in ["holdings_ro.csv", "holdings_eu.csv", "holdings_us.csv"]:
        p = settings.DATA_DIR / fname
        if p.exists():
            d = pd.read_csv(p)
            if "Ticker" not in d.columns:
                continue
            if "Region" not in d.columns:
                # infer from filename
                if "ro" in fname:
                    d["Region"] = "RO"
                elif "eu" in fname:
                    d["Region"] = "EU"
                else:
                    d["Region"] = "US"
            dfs.append(d[["Ticker", "Region"]].dropna().drop_duplicates())
    if not dfs:
        return pd.DataFrame(columns=["Ticker", "Region"])
    return pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _engine_forecast_stub(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Aici presupun că ai deja un engine care produce ER_Pct.
    Dacă OpenAI pică, nu inventăm: returnăm 0.0 și marcăm status.
    """
    out = universe.copy()
    out["ER_Pct"] = 0.0
    out["EngineStatus"] = "FALLBACK"
    out["ModelVersion"] = getattr(settings, "MODEL_VERSION", "StockD")
    out["Notes"] = "fallback neutral"
    return out


def run_weekly_forecast() -> None:
    week_start, week_end = _week_bounds()
    universe = _load_universe()
    if universe.empty:
        send_telegram_message("Weekly forecast: universe is empty (no holdings).")
        return

    # TODO: replace with your real engine call
    raw = _engine_forecast_stub(universe)

    calib = _load_calibration()
    scores = _load_scores()
    mentor_safe = _load_mentor_safe()

    adj = _apply_calibration_and_scoring(raw, calib, scores, mentor_safe)

    # persist in data/forecasts_stockd.csv
    forecasts_path = settings.DATA_DIR / "forecasts_stockd.csv"
    _ensure = forecasts_path.parent
    _ensure.mkdir(parents=True, exist_ok=True)

    # build rows
    today = date.today().isoformat()
    rows = []
    for _, r in adj.iterrows():
        rows.append(
            {
                "Date": today,
                "WeekStart": week_start.isoformat(),
                "TargetDate": week_end.isoformat(),
                "ModelVersion": str(r.get("ModelVersion", "StockD")),
                "Ticker": r["Ticker"],
                "Region": r["Region"],
                "HorizonDays": 5,
                "ER_Pct": float(r["ER_Pct"]),
                "AdjERPct": float(r["AdjERPct"]),
                "Reliability": float(r.get("reliability", 0.5)),
                "Notes": str(r.get("Notes", "")),
            }
        )

    df_out = pd.DataFrame(rows)

    if forecasts_path.exists():
        old = pd.read_csv(forecasts_path)
        df_all = pd.concat([old, df_out], ignore_index=True)
    else:
        df_all = df_out

    df_all.to_csv(forecasts_path, index=False)

    # weekly CSV for Telegram
    weekly_csv = settings.REPORTS_DIR / f"weekly_forecast_{week_start.isoformat()}_{week_end.isoformat()}.csv"
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(weekly_csv, index=False)

    # Telegram message: ALL holdings (not top 10)
    lines = []
    lines.append("StockD weekly forecast")
    lines.append(f"Week: {week_start.isoformat()} → {week_end.isoformat()}")
    lines.append(f"Universe: {len(df_out)} tickers")
    lines.append("")
    lines.append("All signals (AdjERPct):")
    # sort descending by AdjERPct
    srt = df_out.sort_values(["AdjERPct", "ER_Pct"], ascending=[False, False]).reset_index(drop=True)
    for _, r in srt.iterrows():
        lines.append(
            f"- {r['Ticker']} ({r['Region']}): {r['AdjERPct']:+.2f}% (raw {r['ER_Pct']:+.2f}%, rel {r['Reliability']:.2f})"
        )

    send_telegram_message("\n".join(lines))
    send_telegram_document(str(weekly_csv), caption=f"Full forecast list: {week_start.isoformat()} → {week_end.isoformat()}")


if __name__ == "__main__":
    run_weekly_forecast()
