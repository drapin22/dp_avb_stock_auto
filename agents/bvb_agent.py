import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

TICKERS_RO = ["H2O", "SNN", "SNG", "SNP", "TLV", "EL", "ALR", "WINE", "AROBS", "GSH", "HAI"]
DATA_PATH = "data/prices_history.csv"


def fetch_bvb_prices_for_today():
    """
    Ia prețurile de închidere pentru azi pentru toate simbolurile din TICKERS_RO
    de pe pagina de 'historical trading info' a BVB.
    """
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"

    # User-Agent de browser normal, ca să nu fim blocați ca bot
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    # Mai întâi încercăm fără parametru de dată (site-ul ar trebui să dea azi by default)
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"Request to BVB failed: {e}")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if not table:
        print("No table found on BVB page.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    rows = []

    all_tr = table.find_all("tr")
    print(f"Found {len(all_tr)} table rows on BVB page.")

    for tr in all_tr:
        tds = tr.find_all("td")
        if not tds:
            continue

        # text curățat din celule
        cells = [td.get_text(strip=True).replace("\xa0", " ") for td in tds]
        symbol_raw = cells[0]

        # uneori în primul 'td' mai apar spații / descrieri, luăm primul token
        symbol = symbol_raw.split()[0]

        if symbol not in TICKERS_RO:
            continue

        # din restul celulelor extragem toate valorile numerice, în format float
        numeric_vals = []
        for c in cells[1:]:
            # în BVB separatorul zecimal este virgula
            cleaned = c.replace(".", "").replace(",", ".")
            try:
                val = float(cleaned)
            except ValueError:
                continue
            numeric_vals.append(val)

        if not numeric_vals:
            # dacă nu avem niciun numeric, nu putem extrage prețul
            print(f"No numeric values found for {symbol}, row={cells}")
            continue

        # euristica simplă: ultimul numeric din rând = preț de închidere / referință
        close = numeric_vals[-1]

        rows.append(
            {
                "Date": date_str,
                "Ticker": symbol,
                "Close": close,
            }
        )

    if not rows:
        print("Parsed table but found 0 matching tickers from TICKERS_RO.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} rows from BVB for {date_str}: {df['Ticker'].tolist()}")
    return df


def append_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
        combined.to_csv(path, index=False)


def main():
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("No data fetched from BVB for today.")
        return
    append_to_csv(df, DATA_PATH)
    print("Saved prices for", df['Date'].iloc[0], "for", len(df), "tickers.")


if __name__ == "__main__":
    main()
