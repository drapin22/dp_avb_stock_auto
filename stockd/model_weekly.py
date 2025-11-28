from datetime import date, timedelta
import os
from pathlib import Path

import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model  # interfața către modelul tău real


# ----------------------------
# Utilitare de calendar
# ----------------------------

def get_next_monday(d: date) -> date:
    """
    Returnează următoarea zi de luni după data d.
    Dacă azi e deja luni, considerăm luni de săptămâna VIITOARE.
    """
    # Monday = 0, Sunday = 6
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


# ----------------------------
# Încărcare dețineri & prețuri
# ----------------------------

def load_all_holdings() -> pd.DataFrame:
    """
    Combină toate deținerile (RO, EU, US) într-un singur DataFrame.

    Așteaptă fișierele:
      - data/holdings_ro.csv
      - data/holdings_eu.csv
      - data/holdings_us.csv

    Coloane minime: ['Ticker'] + opțional ['Region', 'Active'].
    Dacă lipsește 'Region', folosim default pe baza fișierului.
    Dacă există 'Active', filtrăm la Active == 1.
    """
    sources = [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]

    print("[DEBUG] Loading holdings from:")
    dfs: list[pd.DataFrame] = []

    for path, region_default in sources:
        print(f"  - {path} (default Region={region_default})")
        if not Path(path).exists():
            print(f"    [WARN] File not found, skipping.")
            continue

        df = pd.read_csv(path)
        print(f"    [DEBUG] {path.name}: {len(df)} rows RAW")

        # adăugăm Region dacă lipsește
        if "Region" not in df.columns:
            df["Region"] = region_default

        # filtrăm doar Active == 1 dacă există coloana
        if "Active" in df.columns:
            df = df[df["Active"] == 1]
            print(f"    [DEBUG] {path.name}: {len(df)} rows after filter Active=1")

        dfs.append(df[["Ticker", "Region"]])

    if not dfs:
        print("[WARN] No holdings files found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Ticker", "Region"])

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"[DEBUG] Combined holdings: {len(combined)} rows")
    return combined


def load_prices_history() -> pd.DataFrame:
    """
    Încarcă istoricul de prețuri din data/prices_history.csv.

    Coloane așteptate:
      ['Date', 'Ticker', 'Region', 'Currency', 'Close']
    """
    path = settings.PRICES_HISTORY
    if not Path(path).exists():
        print(f"[WARN] Prices history file not found: {path}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    print(f"[DEBUG] Loaded prices_history.csv: {len(df)} rows")
    return df


# ----------------------------
# Rularea săptămânală a modelului
# ----------------------------

def run_stockd_weekly_model():
    """
    Rulează modelul StockD pentru SĂPTĂMÂNA URMĂTOARE și salvează
    predicțiile într-un fișier CSV de tip benchmark:

      data/forecasts_stockd.csv

    Fiecare rulare adaugă un nou set de rânduri (tickers * o săptămână),
    fără să dubleze intrările deja existente pentru aceeași combinație
    [Date, Ticker, TargetDate, ModelVersion].
    """
    today = date.today()

    # Săptămâna viitoare: luni → vineri
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)  # Monday + 4 = Friday
    horizon_days = (target_date - week_start).days + 1  # e.g. 5 zile

    print(
        f"[MODEL] Today = {today}, "
        f"forecasting week {week_start} – {target_date}"
    )

    # 1. Dețineri
    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    # 2. Istoric de prețuri
    prices_history = load_prices_history()

    # 3. Chemăm modelul real prin interfața standard
    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    if not isinstance(preds, pd.DataFrame):
        raise TypeError("run_stockd_model must return a pandas DataFrame.")

    if "ER_Pct" not in preds.columns:
        raise ValueError("run_stockd_model must return a column named 'ER_Pct'.")

    # 4. Ne asigurăm că avem un merge curat pe Ticker + Region
    merged = holdings.merge(
        preds[["Ticker", "Region", "ER_Pct"]],
        on=["Ticker", "Region"],
        how="left",
    )

    if merged["ER_Pct"].isna().any():
        missing = merged[merged["ER_Pct"].isna()][["Ticker", "Region"]]
        print("[MODEL] WARNING: missing ER_Pct for some tickers:")
        print(missing.to_string(index=False))

    # 5. Construim rândurile de forecast pentru fișierul de benchmark
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Date": today.strftime("%Y-%m-%d"),          # data rulării modelului
                "WeekStart": week_start.strftime("%Y-%m-%d"),
                "TargetDate": target_date.strftime("%Y-%m-%d"),
                "ModelVersion": "StockD_V10.7F+",
                "Ticker": row["Ticker"],
                "Region": row["Region"],
                "HorizonDays": horizon_days,
                "ER_Pct": float(row["ER_Pct"])
                if pd.notna(row["ER_Pct"])
                else None,
                "Notes": "weekly auto-forecast for next week",
            }
        )

    new_df = pd.DataFrame(rows)
    print(f"[MODEL] Generated {len(new_df)} forecast rows.")

    # 6. Salvăm în data/forecasts_stockd.csv
    forecasts_path = settings.FORECASTS_STOCKD
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if forecasts_path.exists():
        old = pd.read_csv(forecasts_path)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined.drop_duplicates(
            subset=["Date", "Ticker", "TargetDate", "ModelVersion"],
            keep="last",
            inplace=True,
        )
        combined.to_csv(forecasts_path, index=False)
    else:
        new_df.to_csv(forecasts_path, index=False)

    print(f"[MODEL] Saved {len(new_df)} weekly forecasts to {forecasts_path}.")


if __name__ == "__main__":
    run_stockd_weekly_model()
