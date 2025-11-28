from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings


def get_week_start(d: date) -> date:
    # luni din săptămâna respectivă
    return d - timedelta(days=d.weekday())


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
    """Aici se leagă modelul tău StockD de benchmark-ul săptămânal."""
    today = date.today()
    week_start = get_week_start(today)

    # pentru claritate, forțăm ca benchmark-ul să fie rulat LUNI
    if today.weekday() != 0:
        print(f"[MODEL] Today is {today}, not Monday. "
              f"Still generating forecasts, WeekStart={week_start}.")
    else:
        print(f"[MODEL] Weekly forecast for week starting {week_start}.")

    # definești orizontul: de ex. 5 zile până vineri
    horizon_days = 5
    target_date = today + timedelta(days=horizon_days)

    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    rows = []

    # 🧠 AICI intră modelul tău real:
    # eu pun un placeholder simplu: ER_Pct = 2% pentru toate
    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        region = row["Region"]

        er_pct = 2.0  # TODO: înlocuiești cu output-ul modelului tău StockD

        rows.append(
            {
                "Date": today.strftime("%Y-%m-%d"),
                "WeekStart": week_start.strftime("%Y-%m-%d"),
                "TargetDate": target_date.strftime("%Y-%m-%d"),
                "ModelVersion": "StockD_V10.7F+",
                "Ticker": ticker,
                "Region": region,
                "HorizonDays": horizon_days,
                "ER_Pct": er_pct,
                "Notes": "weekly auto-forecast",
            }
        )

    new_df = pd.DataFrame(rows)

    forecasts_path = settings.DATA_DIR / "forecasts_stockd.csv"
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    if forecasts_path.exists():
        old = pd.read_csv(forecasts_path)
        combined = pd.concat([old, new_df], ignore_index=True)
        # deduplicăm dacă ai rulat de mai multe ori în aceeași zi pentru aceiași tickeri
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
