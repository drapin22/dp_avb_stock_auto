from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model


def get_next_monday(d: date) -> date:
    """
    Returnează următoarea zi de luni după data d.
    Dacă azi e deja luni, întoarce luni-ul viitor (peste 7 zile).
    """
    # Monday = 0, Sunday = 6
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


def load_all_holdings() -> pd.DataFrame:
    """
    Încarcă toate deținerile din:
      - data/holdings_ro.csv
      - data/holdings_eu.csv
      - data/holdings_us.csv

    și întoarce un DataFrame cu coloanele:
      ['Ticker', 'Region']

    Are loguri [DEBUG] ca să poți verifica ce se întâmplă în GitHub Actions.
    """
    dfs: list[pd.DataFrame] = []

    files = [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]

    print("[DEBUG] Loading holdings from:")
    for path, region_default in files:
        print(f"  - {path} (default Region={region_default})")

    for path, region_default in files:
        if not path.exists():
            print(f"[DEBUG] {path} NOT FOUND, skip.")
            continue

        df = pd.read_csv(path)
        print(f"[DEBUG] {path.name}: {len(df)} rows RAW")

        # Dacă nu avem coloană Region, o completăm cu implicitul
        if "Region" not in df.columns:
            df["Region"] = region_default

        # Dacă există coloana Active, păstrăm doar rândurile Active == 1
        if "Active" in df.columns:
            before = len(df)
            df = df[df["Active"] == 1]
            print(
                f"[DEBUG] {path.name}: {before} → {len(df)} rows after filter Active=1"
            )

        # Dacă după filtrare nu mai e nimic, trecem la următorul fișier
        if df.empty:
            print(f"[DEBUG] {path.name}: EMPTY after filtering, skip.")
            continue

        # Păstrăm doar coloanele necesare
        df_small = df[["Ticker", "Region"]].copy()
        dfs.append(df_small)

    if not dfs:
        print("[DEBUG] No holdings combined – returning EMPTY DataFrame")
        return pd.DataFrame(columns=["Ticker", "Region"])

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"[DEBUG] Combined holdings: {len(combined)} rows")
    return combined


def load_prices_history() -> pd.DataFrame:
    """
    Încarcă istoricul de prețuri (RO + EU + US) din data/prices_history.csv.
    """
    if not settings.PRICES_HISTORY.exists():
        print("[DEBUG] No prices_history.csv found, returning EMPTY DataFrame")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"[DEBUG] Loaded prices_history.csv: {len(df)} rows")
    return df


def run_stockd_weekly_model():
    """
    Rulează modelul StockD pentru săptămâna următoare și salvează
    predicțiile într-un fișier CSV:
      data/forecasts_stockd.csv

    Structura fișierului:
      Date        = data rulării modelului (duminică)
      WeekStart   = luni din săptămâna țintă
      TargetDate  = vineri din săptămâna țintă
      ModelVersion= ex. 'StockD_V10.7F+'
      Ticker
      Region
      HorizonDays = numărul de zile din forecast (ex. 5)
      ER_Pct      = expected return % pe orizont
      Notes       = descriere liberă
    """
    today = date.today()
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)  # luni → vineri
    horizon_days = (target_date - week_start).days + 1  # 5 zile

    print(f"[MODEL] Today = {today}, forecasting week {week_start} – {target_date}")

    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    prices_history = load_prices_history()

    # 🔗 Aici chemăm modelul tău real (definit în stockd.engine.run_stockd_model)
    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    # Ne asigurăm că avem coloana ER_Pct
    if "ER_Pct" not in preds.columns:
        raise ValueError("run_stockd_model must return a column named 'ER_Pct'.")

    # Ne unim cu holdings ca să fim siguri că avem Region/Ticker corect
    merged = holdings.merge(
        preds[["Ticker", "Region", "ER_Pct"]],
        on=["Ticker", "Region"],
        how="left",
    )

    if merged["ER_Pct"].isna().any():
        missing = merged[merged["ER_Pct"].isna()][["Ticker", "Region"]]
        print("[MODEL] WARNING: missing ER_Pct for some tickers:")
        print(missing.to_string(index=False))

    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Date": today.strftime("%Y-%m-%d"),          # când ai rulat modelul
                "WeekStart": week_start.strftime("%Y-%m-%d"),
                "TargetDate": target_date.strftime("%Y-%m-%d"),
                "ModelVersion": "StockD_V10.7F+",
                "Ticker": row["Ticker"],
                "Region": row["Region"],
                "HorizonDays": horizon_days,
                "ER_Pct": float(row["ER_Pct"]),
                "Notes": "weekly auto-forecast for next week",
            }
        )

    new_df = pd.DataFrame(rows)

    forecasts_path = settings.DATA_DIR / "forecasts_stockd.csv"
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
        print(f"[MODEL] Updated forecasts_stockd.csv with {len(new_df)} new rows.")
    else:
        new_df.to_csv(forecasts_path, index=False)
        print(f"[MODEL] Created forecasts_stockd.csv with {len(new_df)} rows.")

    print(f"[MODEL] Saved {len(new_df)} weekly forecasts to {forecasts_path}.")


if __name__ == "__main__":
    run_stockd_weekly_model()
