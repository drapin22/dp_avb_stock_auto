# stockd/run_daily.py
from __future__ import annotations

import argparse
import os
from datetime import date
import pandas as pd
import yfinance as yf

from stockd import settings


def _load_region_tickers(region: str) -> list[str]:
    path_map = {"RO": settings.HOLDINGS_RO, "EU": settings.HOLDINGS_EU, "US": settings.HOLDINGS_US}
    path = path_map[region]
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]
    if "Ticker" not in df.columns:
        return []
    return df["Ticker"].dropna().astype(str).unique().tolist()


def fetch_prices_for_region(region: str) -> pd.DataFrame:
    tickers = _load_region_tickers(region)
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    print(f"[FETCH] {region}: bulk download for {len(tickers)} tickers")
    today = date.today()
    start = today.isoformat()
    end = (today.replace(day=today.day) ).isoformat()  # unused, keep simple

    # yfinance: folosim period=2d ca să prindem “latest available”
    data = yf.download(
        tickers=" ".join(tickers),
        period="2d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []
    currency = "RON" if region == "RO" else ("EUR" if region == "EU" else "USD")

    def extract_close(sym: str) -> float | None:
        try:
            # single ticker
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                s = data["Close"].dropna()
                return float(s.iloc[-1]) if not s.empty else None
            # multi ticker
            s = data[sym]["Close"].dropna()
            return float(s.iloc[-1]) if not s.empty else None
        except Exception:
            return None

    for sym in tickers:
        close = extract_close(sym)
        if close is None:
            print(f"[FETCH] No close for {sym}")
            continue
        rows.append({
            "Date": today.isoformat(),
            "Ticker": sym,
            "Region": region,
            "Currency": currency,
            "Close": close,
        })

    return pd.DataFrame(rows)


def main(regions: list[str]) -> None:
    all_rows = []
    for region in regions:
        region = region.strip().upper()
        if region not in {"RO", "EU", "US"}:
            continue
        df = fetch_prices_for_region(region)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("[DAILY] Nothing fetched.")
        return

    new = pd.concat(all_rows, ignore_index=True)
    dest = settings.PRICES_HISTORY
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if dest.exists():
        old = pd.read_csv(dest)
        combined = pd.concat([old, new], ignore_index=True)
        combined.drop_duplicates(subset=["Date", "Ticker", "Region"], keep="last", inplace=True)
        combined.to_csv(dest, index=False)
    else:
        new.to_csv(dest, index=False)

    print(f"[DAILY] Saved updated prices to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default="EU,US")
    args = parser.parse_args()
    region_list = [x.strip() for x in args.regions.split(",") if x.strip()]
    main(region_list)
