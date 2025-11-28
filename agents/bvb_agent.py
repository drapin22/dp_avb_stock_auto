import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

DATA_PATH = "data/prices_history.csv"
HOLDINGS_RO_PATH = "data/holdings_ro.csv"


def load_ro_tickers():
    if not os.path.exists(HOLDINGS_RO_PATH):
        # fallback dacă nu există fișierul
        return ["H2O", "SNN", "SNG", "SNP", "TLV", "EL",
                "ALR", "WINE", "AROBS", "GSH", "HAI"]

    df = pd.read_csv(HOLDINGS_RO_PATH)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]
    return df["Ticker"].dropna().unique().tolist()


def fetch_bvb_prices_for_today():
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")
    tickers_ro = set(load_ro_tickers())

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"
    params = {"d": date_str}

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    rows = []

    if not table:
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue

        symbol = tds[0]
        if symbol not in tickers_ro:
            continue

        # Coloana "Închidere" este index 9
        try:
            close_str = tds[9].replace(",", ".")
            close = float(close_str)
        except Exception:
            continue

        rows.append({
            "Date": date_str,
            "Ticker": symbol,
            "Region": "RO",
            "Currency": "RON",
            "Close": close,
        })

    return pd.DataFrame(rows)


def append_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["Date", "Ticker", "Region"], keep="last", inplace=True)
        combined.to_csv(path, index=False)


def main():
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("No data fetched from BVB for today.")
        return

    append_to_csv(df, DATA_PATH)
    print("Saved RO prices for:", df["Date"].iloc[0], "Tickers:", df["Ticker"].tolist())


if __name__ == "__main__":
    main()
