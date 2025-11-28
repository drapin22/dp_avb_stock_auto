import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

DATA_PATH = "data/prices_history.csv"
HOLDINGS_EU_PATH = "data/holdings_eu.csv"
HOLDINGS_US_PATH = "data/holdings_us.csv"


def load_holdings(path: str, default_region: str) -> pd.DataFrame:
    """
    Citește fișierul de holdings (EU sau US) și filtrează Active=1.
    """
    if not os.path.exists(path):
        print(f"[YF] Holdings file not found: {path}. Skipping {default_region}.")
        return pd.DataFrame(columns=["Ticker", "Name", "Region", "Currency", "Active"])

    df = pd.read_csv(path)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]
    if "Region" not in df.columns:
        df["Region"] = default_region
    return df


def fetch_yahoo_closes_for_date(symbols, date: datetime.date) -> dict:
    """
    Ia prețul de închidere pentru o listă de simboluri Yahoo în ziua dată.
    Folosește o fereastră [date, date+1] ca să prindă time-zone-uri.
    """
    if not symbols:
        return {}

    start = date
    end = date + timedelta(days=1)

    print(f"[YF] Downloading prices for {symbols} on {date}.")

    data = yf.download(
        tickers=" ".join(symbols),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    closes = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                series = data["Close"]
            else:
                series = data[sym]["Close"]
            val = float(series.dropna().iloc[-1])
            closes[sym] = val
        except Exception:
            print(f"[YF] No close data for {sym} on {date}.")
    return closes


def append_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(
            subset=["Date", "Ticker", "Region"], keep="last", inplace=True
        )
        combined.to_csv(path, index=False)


def main():
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    # 1) Încarcă holdings EU și US
    eu = load_holdings(HOLDINGS_EU_PATH, "EU")
    us = load_holdings(HOLDINGS_US_PATH, "US")

    frames = []
    for region_name, df in [("EU", eu), ("US", us)]:
        if df.empty:
            print(f"[YF] No holdings for region {region_name}.")
            continue

        symbols = df["Ticker"].dropna().unique().tolist()
        closes_map = fetch_yahoo_closes_for_date(symbols, today)

        rows = []
        for _, row in df.iterrows():
            sym = row["Ticker"]
            if sym not in closes_map:
                continue
            rows.append(
                {
                    "Date": date_str,
                    "Ticker": sym,
                    "Region": row.get("Region", region_name),
                    "Currency": row.get("Currency", "EUR" if region_name == "EU" else "USD"),
                    "Close": closes_map[sym],
                }
            )

        if rows:
            sub_df = pd.DataFrame(rows)
            print(
                f"[YF] Fetched {len(sub_df)} prices for region {region_name}: "
                f"{sub_df['Ticker'].tolist()}"
            )
            frames.append(sub_df)
        else:
            print(f"[YF] No prices fetched for region {region_name}.")

    if not frames:
        print("[YF] No data fetched from Yahoo for EU/US today.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    append_to_csv(all_df, DATA_PATH)
    print(
        f"[YF] Saved {len(all_df)} global prices for {date_str} "
        f"into {DATA_PATH}."
    )


if __name__ == "__main__":
    main()
