import os
from datetime import datetime, timedelta

import pandas as pd

DATA_PATH = "data/prices_history.csv"
OUT_PATH = "data/weekly_returns_latest.csv"


def get_week_bounds(today: datetime.date):
    # luăm săptămâna de Luni-Vineri care s-a încheiat vineri trecut
    # dacă rulează sâmbătă sau duminică, tot săptămâna curentă încheiată vineri
    weekday = today.weekday()  # 0=Mon ... 6=Sun
    # offset până la vineri (4)
    delta_to_friday = (weekday - 4) % 7
    last_friday = today - timedelta(days=delta_to_friday)
    monday = last_friday - timedelta(days=4)
    return monday, last_friday


def main():
    if not os.path.exists(DATA_PATH):
        print("[WEEKLY] No prices_history.csv found.")
        return

    today = datetime.utcnow().date()
    start, end = get_week_bounds(today)

    print(f"[WEEKLY] Evaluating week {start} to {end}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    week = df[mask].copy()
    if week.empty:
        print("[WEEKLY] No data for that week.")
        return

    # calculăm preț de start (luni) și de final (vineri) pe fiecare Ticker/Region
    week["Date_only"] = week["Date"].dt.date

    grouped = week.sort_values("Date").groupby(["Region", "Ticker"])

    rows = []
    for (region, ticker), g in grouped:
        first = g.iloc[0]
        last = g.iloc[-1]
        start_price = first["Close"]
        end_price = last["Close"]
        if start_price <= 0:
            continue
        pct = (end_price / start_price - 1) * 100.0

        rows.append(
            {
                "Region": region,
                "Ticker": ticker,
                "StartDate": start,
                "EndDate": end,
                "StartPrice": start_price,
                "EndPrice": end_price,
                "WeekReturn_pct": pct,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["Region", "WeekReturn_pct"], ascending=[True, False], inplace=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"[WEEKLY] Saved weekly returns to {OUT_PATH}")


if __name__ == "__main__":
    main()
