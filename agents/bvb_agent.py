import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

TICKERS_RO = ["H2O","SNN","SNG","SNP","TLV","EL","ALR","WINE","AROBS","GSH","HAI"]
DATA_PATH = "data/prices_history.csv"

def fetch_bvb_prices_for_today():
    today = datetime.utcnow().date()
    # aici poți schimba în funcție de fuseau
    date_str = today.strftime("%Y-%m-%d")

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"
    params = {"d": date_str}  # FORMATUL EXACT poate trebui ajustat

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    rows = []

    if not table:
        return pd.DataFrame(columns=["Date","Ticker","Close"])

    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        symbol = tds[0]
        if symbol in TICKERS_RO:
            # ATENȚIE: indexul coloanei cu prețul de închidere poate fi altul.
            # Când ajungem la testare, îl ajustăm (de ex. tds[5] sau tds[6]).
            close_str = tds[-2].replace(",", ".")
            try:
                close = float(close_str)
            except ValueError:
                continue

            rows.append({
                "Date": date_str,
                "Ticker": symbol,
                "Close": close
            })

    return pd.DataFrame(rows)

def append_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["Date","Ticker"], keep="last", inplace=True)
        combined.to_csv(path, index=False)

def main():
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("No data fetched from BVB for today.")
        return
    append_to_csv(df, DATA_PATH)
    print("Saved prices for", df["Date"].iloc[0], "for", len(df), "tickers.")

if __name__ == "__main__":
    main()
