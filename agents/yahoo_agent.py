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


def _extract_close_from_download(data, sym: str):
    """
    Extrage coloana 'Close' pentru un simbol din rezultatul yf.download.
    Funcționează și pentru un singur ticker (Series) și pentru multi-ticker.
    """
    if data is None or len(data) == 0:
        return None

    try:
        # un singur ticker -> DataFrame cu coloană Close
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            series = data["Close"]
        else:
            # multi-ticker -> data[sym]["Close"]
            series = data[sym]["Close"]

        series = series.dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])
    except Exception:
        return None


def fetch_yahoo_closes_for_date(symbols, date: datetime.date) -> dict:
    """
    Ia prețul de închidere pentru o listă de simboluri Yahoo în ziua dată.
    1) încearcă un download bulk
    2) pentru fiecare simbol care iese NaN, încearcă un download separat
    """
    if not symbols:
        return {}

    start = date
    end = date + timedelta(days=1)

    print(f"[YF] Bulk downloading prices for {symbols} on {date}.")

    bulk = yf.download(
        tickers=" ".join(symbols),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    closes = {}

    for sym in symbols:
        # 1) încercăm să extragem din bulk
        close_val = _extract_close_from_download(bulk, sym)

        # 2) dacă nu există, facem request separat
        if close_val is None:
            print(f"[YF] No close in bulk for {sym}, trying single download.")
            try:
                single = yf.download(
                    tickers=sym,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    progress=False,
                )
                close_val = _extract_close_from_download(single, sym)
            except Exception as e:
                print(f"[YF] Error downloading {sym} individually: {e}")
                close_val = None

        if close_val is None:
            print(f"[YF] No close data for {sym} on {date}.")
        else:
            closes[sym] = close_val

    return closes


def append_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Adaugă noile prețuri în prices_history.csv, cu deduplicare
    pe (Date, Ticker, Region).
    """
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

    # 2) Încarcă istoricul existent (pentru carry-forward)
    history = None
    if os.path.exists(DATA_PATH):
        history = pd.read_csv(DATA_PATH)

    frames = []

    # 3) Procesează EU și US
    for region, df in [("EU", eu), ("US", us)]:
        if df.empty:
            print(f"[YF] No holdings for region {region}.")
            continue

        symbols = df["Ticker"].dropna().unique().tolist()
        closes_map = fetch_yahoo_closes_for_date(symbols, today)

        rows = []
        for _, row in df.iterrows():
            sym = row["Ticker"]
            currency = row.get("Currency", "EUR" if region == "EU" else "USD")

            # A) Yahoo a dat preț pentru azi
            if sym in closes_map:
                close_value = closes_map[sym]
            else:
                # B) Yahoo a eșuat → încercăm carry-forward din history
                close_value = None
                if history is not None:
                    prev = history[history["Ticker"] == sym]
                    if not prev.empty:
                        close_value = float(prev.iloc[-1]["Close"])
                        print(f"[YF] Carry-forward for {sym}: {close_value}")

                if close_value is None:
                    print(f"[YF] No data for {sym} at all. Skipping.")
                    continue

            rows.append(
                {
                    "Date": date_str,
                    "Ticker": sym,
                    "Region": region,
                    "Currency": currency,
                    "Close": close_value,
                }
            )

        if rows:
            sub_df = pd.DataFrame(rows)
            print(
                f"[YF] Saved {len(sub_df)} prices for region {region}: "
                f"{sub_df['Ticker'].tolist()}"
            )
            frames.append(sub_df)
        else:
            print(f"[YF] No prices saved for region {region}.")

    if not frames:
        print("[YF] No data saved for EU/US today.")
        return

    full_df = pd.concat(frames, ignore_index=True)
    append_to_csv(full_df, DATA_PATH)
    print(
        f"[YF] Saved {len(full_df)} global prices into {DATA_PATH} "
        f"for {date_str}."
    )


if __name__ == "__main__":
    main()
