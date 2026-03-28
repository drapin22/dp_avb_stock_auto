"""
scripts/backfill_prices.py

Downloads 3 years of daily OHLCV data for all holdings using yfinance.
Merges with existing prices_history.csv (no duplicates).
Also downloads Volume so volume_ratio feature works.

Usage:
    python -m scripts.backfill_prices
    # or directly:
    python scripts/backfill_prices.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Make sure stockd package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from stockd import settings


# Ticker mapping: yfinance symbol -> (our_ticker, region, currency)
# BVB tickers need .RO suffix on yfinance; some need manual mapping
TICKER_MAP = {
    # RO - BVB
    "TLV.RO":   ("TLV",  "RO", "RON"),
    "SNN.RO":   ("SNN",  "RO", "RON"),
    "SNP.RO":   ("SNP",  "RO", "RON"),
    "FP.RO":    ("FP",   "RO", "RON"),
    "WINE.RO":  ("WINE", "RO", "RON"),
    "H2O.RO":   ("H2O",  "RO", "RON"),
    "AROBS.RO": ("AROBS","RO", "RON"),
    "AQ.RO":    ("AQ",   "RO", "RON"),
    "EL.RO":    ("EL",   "RO", "RON"),
    "AG.RO":    ("AG",   "RO", "RON"),
    "TRP.RO":   ("TRP",  "RO", "RON"),
    "ALR.RO":   ("ALR",  "RO", "RON"),
    "GSH.RO":   ("GSH",  "RO", "RON"),
    "HAI.RO":   ("HAI",  "RO", "RON"),
    "NRF.RO":   ("NRF",  "RO", "RON"),
    # US
    "BABA":     ("BABA", "US", "USD"),
    "NIO":      ("NIO",  "US", "USD"),
    "PATH":     ("PATH", "US", "USD"),
    "JD":       ("JD",   "US", "USD"),
    "NVDA":     ("NVDA", "US", "USD"),
    # EU
    "BATE.DE":  ("BATE.DE", "EU", "EUR"),
    "LHA.DE":   ("LHA.DE",  "EU", "EUR"),
}

LOOKBACK_YEARS = 3


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance MultiIndex columns (new API)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_ticker(yf_sym: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        raw = yf.download(yf_sym, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        raw = _flatten_multiindex(raw)
        if "Close" not in raw.columns:
            return None
        raw = raw.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
        return raw[["Date"] + [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]]
    except Exception as e:
        print(f"  ERROR downloading {yf_sym}: {e}")
        return None


def backfill():
    end   = date.today()
    start = end - timedelta(days=LOOKBACK_YEARS * 365 + 30)

    print(f"Downloading {len(TICKER_MAP)} tickers from {start} to {end}...")

    rows = []
    failed = []
    for yf_sym, (our_ticker, region, currency) in TICKER_MAP.items():
        print(f"  {yf_sym} -> {our_ticker} ({region})...", end=" ", flush=True)
        df = download_ticker(yf_sym, start, end)
        if df is None or df.empty:
            print("FAILED")
            failed.append(yf_sym)
            continue
        print(f"{len(df)} rows")
        for _, r in df.iterrows():
            row = {
                "Date":     str(r["Date"]),
                "Ticker":   our_ticker,
                "Region":   region,
                "Currency": currency,
                "Close":    round(float(r["Close"]), 6) if not pd.isna(r["Close"]) else None,
            }
            if "Volume" in r and not pd.isna(r["Volume"]):
                row["Volume"] = int(r["Volume"])
            rows.append(row)

    if not rows:
        print("No data downloaded. Check yfinance and ticker symbols.")
        return

    new_df = pd.DataFrame(rows).dropna(subset=["Close"])
    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime("%Y-%m-%d")

    # Merge with existing
    out_path = settings.PRICES_HISTORY
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Add Volume column to existing if missing
        if "Volume" not in existing.columns:
            existing["Volume"] = None
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["Date","Ticker","Region"], keep="last", inplace=True)
        combined.sort_values(["Ticker","Region","Date"], inplace=True)
        combined.to_csv(out_path, index=False)
        print(f"\nMerged: {len(combined)} total rows saved to {out_path}")
    else:
        new_df.sort_values(["Ticker","Region","Date"], inplace=True)
        new_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(new_df)} rows to {out_path}")

    if failed:
        print(f"\nFailed tickers ({len(failed)}): {', '.join(failed)}")
        print("BVB tickers may need .RO suffix or may not be on yfinance.")
        print("Check: https://finance.yahoo.com and search ticker manually.")

    print("\nDone. Run weekly forecast to retrain with new history.")


if __name__ == "__main__":
    backfill()
