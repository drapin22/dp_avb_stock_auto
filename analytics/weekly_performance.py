# analytics/weekly_performance.py
import pandas as pd
from stockd import settings


def compute_weekly_performance(region_filter=None) -> pd.DataFrame:
    path = settings.PRICES_HISTORY
    if not path.exists():
        raise FileNotFoundError(f"No prices file found at {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if region_filter is not None:
        df = df[df["Region"] == region_filter]

    df = df.sort_values(["Ticker", "Region", "Date"])

    df["DailyReturn"] = df.groupby(["Ticker", "Region"])["Close"].pct_change()
    df["WeekStart"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["CumReturnFromWeekStart"] = (
        (1 + df["DailyReturn"])
        .groupby([df["Ticker"], df["Region"], df["WeekStart"]])
        .cumprod()
        - 1
    )
    return df


def main():
    df = compute_weekly_performance()
    print(df.tail(20))


if __name__ == "__main__":
    main()
