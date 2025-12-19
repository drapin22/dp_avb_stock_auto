from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

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
    "DXY": "DX-Y.NYB",     # dollar index (Yahoo)
    "EURUSD": "EURUSD=X",
    "EURRON": "EURRON=X",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "US10Y": "^TNX",       # yield proxy (în puncte, nu % return normal)
}


def _download_close(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    if yf is None:
        return None
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col is None:
            return None
        s = df[col].dropna()
        s.index = pd.to_datetime(s.index)
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


def get_macro_snapshot(as_of: datetime, cfg: Optional[MacroConfig] = None) -> Dict:
    cfg = cfg or MacroConfig()
    start = as_of - timedelta(days=cfg.lookback_days)
    end = as_of + timedelta(days=1)

    snap: Dict[str, Dict] = {"as_of": as_of.date().isoformat(), "series": {}}

    for k, t in DEFAULT_SERIES.items():
        s = _download_close(t, start=start, end=end)
        if s is None:
            snap["series"][k] = {"ticker": t, "last": None, "ret_5d_pct": None}
            continue

        last = float(s.iloc[-1])
        ret5 = _pct_return(s, cfg.ret_window_days)
        snap["series"][k] = {"ticker": t, "last": last, "ret_5d_pct": ret5}

    # region drift proxies (weekly “market mood”)
    snap["regions"] = {
        "US": {"bench": "SPX", "bench_ret_5d_pct": snap["series"]["SPX"]["ret_5d_pct"]},
        "EU": {"bench": "STOXX50", "bench_ret_5d_pct": snap["series"]["STOXX50"]["ret_5d_pct"]},
        "RO": {"bench": "EURRON", "bench_ret_5d_pct": 0.0},  # RO drift îl facem în features.py din proxy intern
    }

    return snap
