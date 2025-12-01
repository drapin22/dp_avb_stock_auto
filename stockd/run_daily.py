# stockd/run_daily.py
import argparse
import pandas as pd
import os
from datetime import date
import yfinance as yf

from stockd import settings


def fetch_prices_for_region(region: str) -> pd.DataFrame:
    """
    Fetch daily prices for a specific region.
    """
    path_map = {
        "RO": settings.HOLDINGS_RO,
        "EU": settings.HOLDINGS_EU,
        "US": settings.HOLDINGS_US,
    }

    if region not in path_map:
        raise ValueError(f"Invalid region {region}")

    path = path_map[region]
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]

    tickers = df["Ticker"].tolist()
    if not tickers:
        return pd.DataFrame()

    rows = []
    print(f"[FETCH] Fetching prices for region {region}: {tickers}")

    for tkr in tickers:
        try:
            y = yf.Ticker(tkr)
            hist = y.history(period="1d")
            if hist.empty:
                continue
            close = hist["Close"].iloc[-1]
            rows.append({
                "Date": date.today().isoformat(),
                "Ticker": tkr,
                "Region": region,
                "Currency": "RON" if region == "RO" else "EUR" if region == "EU" else "USD",
                "Close": float(close),
            })
        except Exception as exc:
            print(f"[ERROR] Could not fetch {tkr}: {exc}")

    return pd.DataFrame(rows)


def main(regions):
    print(f"[DAILY] Fetching prices for regions: {regions}")

    all_rows = []

    for region in regions:
        df = fetch_prices_for_region(region)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("[DAILY] Nothing fetched.")
        return

    new = pd.concat(all_rows, ignore_index=True)

    # merge with existing
    dest = settings.DATA_DIR / "prices_history.csv"
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if dest.exists():
        old = pd.read_csv(dest)
        combined = pd.concat([old, new], ignore_index=True)
        combined.drop_duplicates(
            subset=["Date", "Ticker", "Region"],
            keep="last",
            inplace=True,
        )
        combined.to_csv(dest, index=False)
    else:
        new.to_csv(dest, index=False)

    print(f"[DAILY] Saved updated prices to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default="RO,EU,US")
    args = parser.parse_args()

    region_list = [x.strip() for x in args.regions.split(",")]
    main(region_list)
