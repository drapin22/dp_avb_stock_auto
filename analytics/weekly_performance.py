import pandas as pd
from pathlib import Path

from stockd import settings


def compute_weekly_performance(region_filter=None) -> pd.DataFrame:
    """
    Construiește un tabel simplu cu:
    - Date
    - Ticker
    - Close
    - DailyReturn (%)
    - WeekStart (luni din săptămâna respectivă)
    - CumReturnFromWeekStart (%)

    region_filter: "RO", "EU", "US" sau None (toate)
    """
    path = settings.PRICES_HISTORY
    if not path.exists():
        raise FileNotFoundError(f"No prices file found at {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    if region_filter is not None:
        df = df[df["Region"] == region_filter]

    df = df.sort_values(["Ticker", "Date"])

    # Daily % change per ticker
    df["DailyReturn"] = df.groupby("Ticker")["Close"].pct_change()

    # Week start (luni)
    df["WeekStart"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")

    # Cumulative return from Monday per week per ticker
    df["CumReturnFromWeekStart"] = (
        (1 + df["DailyReturn"])
        .groupby([df["Ticker"], df["WeekStart"]])
        .cumprod()
        - 1
    )

    return df


def main():
    df = compute_weekly_performance()
    # Print doar ultimele zile ca sanity check
    print(df.tail(20))


if __name__ == "__main__":
    main()
