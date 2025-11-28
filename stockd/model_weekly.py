from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings


def get_next_monday(d: date) -> date:
    """
    Returnează luni-ul următor.
    Dacă azi e duminică, next_monday = mâine.
    Dacă azi e marți, next_monday = luni de săptămâna viitoare.
    """
    # weekday: Monday=0, Sunday=6
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


def load_all_holdings() -> pd.DataFrame:
    """Combină RO + EU + US ca să ai lista completă de tickere pentru forecast."""
    dfs = []
    for path, region_default in [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Region" not in df.columns:
            df["Region"] = region_default
        if "Active" in df.columns:
            df = df[df["Active"] == 1]
        dfs.append(df[["Ticker", "Region"]])

    if not dfs:
        return pd.DataFrame(columns=["Ticker", "Region"])

    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def run_stockd_weekly_model():
    """
    Rulezi asta DUMINICĂ.
    - Date        = azi (duminică)  -> când a rulat modelul
    - WeekStart   = luni viitoare   -> săptămâna pentru care facem forecast
    - TargetDate  = vineri viitoare -> benchmark pe 5 zile
    """
    today = date.today()
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)  # luni + 4 = vineri

    print(f"[MODEL] Today = {today}, forecasting week {week_start} – {target_date}")

    horizon_days = (target_date - week_start).days + 1  # 5 zile (L–V)

    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    rows = []

    # TODO: aici bagi modelul tău real.
    # Momentan pun un placeholder: +2% pentru toate, ca să fie clar formatul.
    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        region = row["Region"]

        er_pct = 2.0  # înlocuiești cu output-ul StockD pentru ticker + regiune

        rows.append(
            {
                "Date": today.strftime("%Y-%m-%d"),          # când a rulat modelul
                "WeekStart": week_start.strftime("%Y-%m-%d"),
                "TargetDate": target_date.strftime("%Y-%m-%d"),
                "ModelVersion": "StockD_V10.7F+",
                "Ticker": ticker,
                "Region": region,
                "HorizonDays": horizon_days,
                "ER_Pct": er_pct,
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
    else:
        new_df.to_csv(forecasts_path, index=False)

    print(f"[MODEL] Saved {len(new_df)} weekly forecasts to {forecasts_path}.")


if __name__ == "__main__":
    run_stockd_weekly_model()
