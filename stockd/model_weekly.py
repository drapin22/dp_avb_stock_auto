from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model   # <- aici legăm cu modelul tău


def get_next_monday(d: date) -> date:
    # Monday=0, Sunday=6
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


def load_all_holdings() -> pd.DataFrame:
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


def load_prices_history() -> pd.DataFrame:
    if not settings.PRICES_HISTORY.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def run_stockd_weekly_model():
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

    # 🔗 Aici chemăm modelul tău real
    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    # Ne asigurăm că avem toate coloanele de care avem nevoie
    if "ER_Pct" not in preds.columns:
        raise ValueError("run_stockd_model must return a column named 'ER_Pct'.")

    # ne unim cu holdings ca să fim siguri că avem Region/Ticker corect
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
    else:
        new_df.to_csv(forecasts_path, index=False)

    print(f"[MODEL] Saved {len(new_df)} weekly forecasts to {forecasts_path}.")


if __name__ == "__main__":
    run_stockd_weekly_model()
