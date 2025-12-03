import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os

REPORT_DIR = "data/weekly_report"
FORECAST_FILE = "data/forecasts_stockd.csv"
PRICES_FILE = "data/prices_history.csv"

os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    forecast = pd.read_csv(FORECAST_FILE)
    prices = pd.read_csv(PRICES_FILE)
    prices["Date"] = pd.to_datetime(prices["Date"])
    return forecast, prices

def compute_weekly_returns(prices, start_date, end_date):
    tickers = prices["Ticker"].unique()
    data = []

    for t in tickers:
        df = prices[prices["Ticker"] == t].sort_values("Date")
        w_open = df[df["Date"] == start_date]
        w_close = df[df["Date"] == end_date]

        if w_open.empty or w_close.empty:
            continue

        open_p = float(w_open["Close"].iloc[0])
        close_p = float(w_close["Close"].iloc[0])
        ret = (close_p - open_p) / open_p * 100

        region = df["Region"].iloc[0]

        data.append({
            "Ticker": t,
            "Region": region,
            "Open": open_p,
            "Close": close_p,
            "Real_Return_Pct": ret
        })

    return pd.DataFrame(data)

def generate_plot(df, output_path):
    df_sorted = df.sort_values("Error_Pct")

    plt.figure(figsize=(12, 8))
    plt.barh(df_sorted["Ticker"], df_sorted["Error_Pct"])
    plt.title("Prediction Error vs Real")
    plt.xlabel("Error (pct points)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_report():
    today = date.today()
    last_sunday = today - timedelta(days=today.weekday() + 1)
    week_start = last_sunday + timedelta(days=1)
    week_end = week_start + timedelta(days=4)

    print(f"[REPORT] Using forecast week = {week_start} -> {week_end}")

    forecast, prices = load_data()

    last_forecast = forecast[forecast["WeekStart"] == str(week_start)]
    if last_forecast.empty:
        raise ValueError("No forecasts found for this week")

    returns = compute_weekly_returns(
        prices,
        pd.to_datetime(week_start),
        pd.to_datetime(week_end)
    )

    merged = last_forecast.merge(returns, on=["Ticker", "Region"], how="left")
    merged["Error_Pct"] = merged["Real_Return_Pct"] - merged["ER_Pct"]

    out_csv = f"{REPORT_DIR}/weekly_report_{week_end}.csv"
    merged.to_csv(out_csv, index=False)

    out_png = f"{REPORT_DIR}/weekly_plot_{week_end}.png"
    generate_plot(merged, out_png)

    print(f"[REPORT] Saved CSV to {out_csv}")
    print(f"[REPORT] Saved plot to {out_png}")

    summary = merged[["Ticker", "ER_Pct", "Real_Return_Pct", "Error_Pct"]] \
        .sort_values("Error_Pct")

    text_summary = summary.to_string(index=False)
    summary_file = f"{REPORT_DIR}/summary_{week_end}.txt"

    with open(summary_file, "w") as f:
        f.write(text_summary)

    print(f"[REPORT] Saved summary to {summary_file}")

    return summary_file

if __name__ == "__main__":
    run_report()
