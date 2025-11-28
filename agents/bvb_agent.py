import os
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_PATH = "data/prices_history.csv"
HOLDINGS_RO_PATH = "data/holdings_ro.csv"


def load_ro_tickers():
    """
    Citește lista de companii românești urmărite din holdings_ro.csv.
    Dacă fișierul nu există, folosește o listă fallback hard-codată.
    """
    if not os.path.exists(HOLDINGS_RO_PATH):
        print("[RO] holdings_ro.csv not found, using fallback list.")
        return [
            "WINE", "EL", "AG", "H2O", "SNN", "AQ", "SNP",
            "TRP", "AROBS", "FP", "TLV", "ALR", "GSH", "HAI", "NRF",
        ]

    df = pd.read_csv(HOLDINGS_RO_PATH)
    if "Active" in df.columns:
        df = df[df["Active"] == 1]
    tickers = df["Ticker"].dropna().unique().tolist()
    print(f"[RO] Loaded {len(tickers)} tickers from holdings_ro.csv: {tickers}")
    return tickers


def fetch_bvb_prices_for_today() -> pd.DataFrame:
    """
    Ia prețurile de închidere (coloana 'Închidere') pentru tickerele din holdings_ro
    de pe pagina de 'Sumar tranzacționare' BVB (m.bvb.ro).
    """
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    tickers_ro = set(load_ro_tickers())
    if not tickers_ro:
        print("[RO] No tickers loaded, aborting.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"
    params = {"d": date_str}

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[RO] HTTP request to BVB failed: {e}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("[RO] No table found on BVB page.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    rows = []
    trs = table.find_all("tr")
    print(f"[RO] Found {len(trs)} rows in BVB table.")

    for tr in trs:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue

        symbol = tds[0]
        if symbol not in tickers_ro:
            continue

        # Structura de pe BVB (desktop/mobil) din screenshot:
        # 0 Simbol
        # 1 Piața
        # 2 Nr. tranz.
        # 3 Volum
        # 4 Valoare
        # 5 Desch.
        # 6 Min.
        # 7 Max.
        # 8 Închidere   ← asta ne interesează
        # 9 Mediu       (sau invers în unele layout-uri)
        # 10 Pret ref.
        # 11 Var (%)
        #
        # În setup-ul actual, indexul 8 produce prețurile corecte confirmate de tine.

        if len(tds) <= 8:
            print(f"[RO] Not enough columns for {symbol}: {tds}")
            continue

        raw_close = tds[8]
        try:
            close = float(raw_close.replace(".", "").replace(",", "."))
        except ValueError:
            print(f"[RO] Could not parse close for {symbol}: '{raw_close}'")
            continue

        rows.append(
            {
                "Date": date_str,
                "Ticker": symbol,
                "Region": "RO",
                "Currency": "RON",
                "Close": close,
            }
        )

    if not rows:
        print("[RO] Parsed BVB table but found 0 matching tickers.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.DataFrame(rows)
    print(
        f"[RO] Fetched {len(df)} rows for {date_str}: "
        f"{df['Ticker'].tolist()}"
    )
    return df


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


def main() -> None:
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("[RO] No data fetched from BVB for today.")
        return

    append_to_csv(df, DATA_PATH)
    print(
        f"[RO] Saved RO prices for {df['Date'].iloc[0]} "
        f"for {len(df)} tickers."
    )


if __name__ == "__main__":
    main()
