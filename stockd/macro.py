from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@dataclass
class MacroConfig:
    lookback_days: int = 180
    ret_window_days: int = 5


DEFAULT_SERIES = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "STOXX50": "^STOXX50E",
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "EURRON": "EURRON=X",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "US10Y": "^TNX",
}


def _to_datetime(x: Union[date, datetime]) -> datetime:
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime.combine(x, datetime.min.time())
    # fallback (shouldn't happen)
    return datetime.utcnow()


def _download_close(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    if yf is None:
        return None
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        # yfinance >=0.2 returns MultiIndex columns — flatten them first
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col is None:
            return None
        s = df[col]
        # If still a DataFrame (can happen with multi-ticker downloads), take first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _pct_return(s: pd.Series, window: int) -> float:
    s = s.dropna()
    if s.shape[0] < window + 1:
        return 0.0
    a = float(s.iloc[-(window + 1)])
    b = float(s.iloc[-1])
    if a == 0:
        return 0.0
    return (b / a - 1.0) * 100.0


def get_macro_snapshot(as_of: Union[date, datetime], cfg: Optional[MacroConfig] = None) -> Dict:
    """
    Returns a snapshot dict and also flattens key fields commonly used by engine:
      vix_level, dxy_ret_5d, oil_ret_5d, gold_ret_5d, bench_ret_5d
    """
    cfg = cfg or MacroConfig()
    as_dt = _to_datetime(as_of)

    start = as_dt - timedelta(days=cfg.lookback_days)
    end = as_dt + timedelta(days=1)

    snap: Dict[str, Dict] = {"as_of": as_dt.date().isoformat(), "series": {}, "regions": {}}

    for k, t in DEFAULT_SERIES.items():
        s = _download_close(t, start=start, end=end)
        if s is None:
            snap["series"][k] = {"ticker": t, "last": None, "ret_5d_pct": None}
            continue

        last = float(s.iloc[-1])
        ret5 = _pct_return(s, cfg.ret_window_days)
        snap["series"][k] = {"ticker": t, "last": last, "ret_5d_pct": ret5}

    # region benchmarks (5d)
    spx5 = snap["series"].get("SPX", {}).get("ret_5d_pct")
    stx5 = snap["series"].get("STOXX50", {}).get("ret_5d_pct")

    snap["regions"] = {
        "US": {"bench": "SPX", "bench_ret_5d_pct": float(spx5) if spx5 is not None else 0.0},
        "EU": {"bench": "STOXX50", "bench_ret_5d_pct": float(stx5) if stx5 is not None else 0.0},
        "RO": {"bench": "RO_PROXY", "bench_ret_5d_pct": 0.0},
    }

    # Flatten common keys used by engine (so engine can do macro.get("vix_level"))
    snap["vix_level"] = float(snap["series"].get("VIX", {}).get("last") or 0.0)
    snap["dxy_ret_5d"] = float(snap["series"].get("DXY", {}).get("ret_5d_pct") or 0.0)
    snap["oil_ret_5d"] = float(snap["series"].get("OIL", {}).get("ret_5d_pct") or 0.0)
    snap["gold_ret_5d"] = float(snap["series"].get("GOLD", {}).get("ret_5d_pct") or 0.0)

    # default bench_ret_5d as US benchmark; engine can override per region if needed
    snap["bench_ret_5d"] = float(snap["regions"]["US"]["bench_ret_5d_pct"] or 0.0)

    return snap
