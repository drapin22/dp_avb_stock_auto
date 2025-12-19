import os
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_PATH = "data/prices_history.csv"
HOLDINGS_RO_PATH = "data/holdings_ro.csv"


def load_ro_tickers():
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
    today = datetime.utcnow().date()
    date_str = today.strftime("%Y-%m-%d")

    tickers_ro = set(load_ro_tickers())
    if not tickers_ro:
        print("[RO] No tickers loaded, aborting.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    url = "https://m.bvb.ro/tradingandstatistics/trading/historicaltradinginfo"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
            "Mobile/15E148 Safari/604.1"
        )
    }

    try:
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        print(f"[RO] HTTP request to BVB failed: {e}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("[RO] No table found on BVB page.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    headers_text = []
    header_tr = table.find("tr")
    if header_tr:
        ths = header_tr.find_all("th")
        if ths:
            headers_text = [th.get_text(strip=True) for th in ths]

    def _idx(name_variants, default_idx: int | None = None) -> int | None:
        if not headers_text:
            return default_idx
        for i, h in enumerate(headers_text):
            for v in name_variants:
                if v.lower() in h.lower():
                    return i
        return default_idx

    idx_symbol = _idx(["simbol", "symbol"], 0)
    idx_close = _idx(["închidere", "inchidere", "close"], 8)

    last_close_map = {}
    if os.path.exists(DATA_PATH):
        try:
            hist = pd.read_csv(DATA_PATH)
            hist = hist[hist.get("Region", "") == "RO"]
            hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
            hist["Ticker"] = hist["Ticker"].astype(str).str.upper().str.strip()
            hist["Close"] = pd.to_numeric(hist.get("Close"), errors="coerce")
            hist = hist.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"])
            last = hist.groupby("Ticker", as_index=False).tail(1)
            last_close_map = {r["Ticker"]: float(r["Close"]) for _, r in last.iterrows()}
        except Exception:
            last_close_map = {}

    rows = []
    trs = table.find_all("tr")
    print(f"[RO] Found {len(trs)} rows in BVB table.")

    for tr in trs:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue

        if idx_symbol is None or idx_symbol >= len(tds):
            continue
        symbol = tds[idx_symbol]
        if symbol not in tickers_ro:
            continue

        if idx_close is None or idx_close >= len(tds):
            print(f"[RO] Close column not found for {symbol}: {tds}")
            continue

        raw_close = tds[idx_close]
        cleaned = raw_close.replace(".", "").replace(",", ".")
        try:
            close = float(cleaned)
        except ValueError:
            print(f"[RO] Could not parse close for {symbol}: '{raw_close}'")
            continue

        if close <= 0 or close > 10_000:
            print(f"[RO] Suspicious close for {symbol}: {close} (raw '{raw_close}')")
            continue

        prev = last_close_map.get(symbol)
        if prev and prev > 0:
            ratio = close / prev
            if ratio > 2.5 or ratio < 0.4:
                print(f"[RO] Suspicious jump for {symbol}: prev {prev} -> {close} (ratio {ratio:.2f}); skipping")
                continue

        rows.append(
            {"Date": date_str, "Ticker": symbol, "Region": "RO", "Currency": "RON", "Close": close}
        )

    if not rows:
        print("[RO] Parsed BVB table but found 0 matching tickers.")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.DataFrame(rows)
    print(f"[RO] Fetched {len(df)} rows for {date_str}: {df['Ticker'].tolist()}")
    return df


def append_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined.drop_duplicates(subset=["Date", "Ticker", "Region"], keep="last", inplace=True)
        combined.to_csv(path, index=False)


def main() -> None:
    df = fetch_bvb_prices_for_today()
    if df.empty:
        print("[RO] No data fetched from BVB for today.")
        return
    append_to_csv(df, DATA_PATH)
    print(f"[RO] Saved RO prices for {df['Date'].iloc[0]} for {len(df)} tickers.")


if __name__ == "__main__":
    main()
