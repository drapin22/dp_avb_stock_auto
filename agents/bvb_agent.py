import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

# Ticker-ele românești pe care vrei să le urmărești
TICKERS_RO = ["H2O", "SNN", "SNG", "SNP", "TLV", "EL",
              "ALR", "WINE", "AROBS", "GSH", "HAI"]

DATA_PATH = "data/prices_history.csv"


def fetch_bvb_prices_for_today() -> pd.DataFrame:
    """
    Ia prețurile de închidere (coloana 'Închidere') pentru tickerele din TICKERS_RO
    de pe pagina de 'Sumar tranzacționare' BVB (varianta mobil/desktop).
    """
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[BVB] HTTP request failed: {e}")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    soup = BeautifulSoup(resp.text, "html.parser")

    # Luăm primul tabel mare – este cel cu "Simbol / Piața / Nr. tranz. / Volum / ... / Închidere / Pret ref / Var (%)"
    table = soup.find("table")
    if not table:
        print("[BVB] No table found on page.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    rows = []
    all_tr = table.find_all("tr")
    print(f"[BVB] Found {len(all_tr)} table rows.")

    for tr in all_tr:
        tds = tr.find_all("td")
        if not tds:
            continue

        # Extragem textul din celule
        cells = [td.get_text(strip=True).replace("\xa0", " ") for td in tds]

        # Ne asigurăm că avem destule coloane (Simbol + cel puțin 11 coloane de numeric)
        if len(cells) < 11:
            continue

        symbol_raw = cells[0]
        symbol = symbol_raw.split()[0]  # în caz că apar spații / extra text

        if symbol not in TICKERS_RO:
            continue

        # Conform capturii BVB:
        # 0 = Simbol
        # 1 = Piața
        # 2 = Nr. tranz.
        # 3 = Volum
        # 4 = Valoare
        # 5 = Desch.
        # 6 = Min.
        # 7 = Max.
        # 8 = Mediu
        # 9 = Închidere  ← asta vrem
        # 10 = Pret ref.
        # 11 = Var (%)

        raw_close = cells[9]
        # Curățăm formatul românesc: punct pentru mii, virgulă pentru zecimale
        cleaned = raw_close.replace(".", "").replace(",", ".")
        try:
            close = float(cleaned)
        except ValueError:
            print(f"[BVB] Could not parse close for {symbol}: '{raw_close}'")
            continue

        rows.append(
            {
                "Date": date_str,
                "Ticker": symbol,
                "Close": close,
            }
        )

    if not rows:
        print("[BVB] Parsed table but found 0 matching tickers from TICKERS_RO.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    df = pd.DataFrame(rows)
    print(f"[BVB] Fetched {len(df)} rows for {date_str}: {df['Ticker'].tolist()}")
    return df


def append_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
        combined.to_csv(path, index=False)


def main() -> None:
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("[BVB] No data fetched from BVB for today.")
        return
    append_to_csv(df, DATA_PATH)
    print(f"[BVB] Saved prices for {df['Date'].iloc[0]} for {len(df)} tickers.")


if __name__ == "__main__":
    main()
