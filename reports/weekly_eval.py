from __future__ import annotations

from datetime import date
import os
import pandas as pd

from stockd import settings


def load_prices_history() -> pd.DataFrame:
    path = settings.PRICES_HISTORY
    if not path.exists():
        print(f"[EVAL] No prices_history.csv at {path}")
        return pd.DataFrame(columns=["Date", "Ticker", "Region", "Currency", "Close"])

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_forecasts() -> pd.DataFrame:
    path = settings.DATA_DIR / "forecasts_stockd.csv"
    if not path.exists():
        print(f"[EVAL] No forecasts_stockd.csv at {path}")
        return pd.DataFrame(
            columns=[
                "Date",
                "WeekStart",
                "TargetDate",
                "ModelVersion",
                "Ticker",
                "Region",
                "HorizonDays",
                "ER_Pct",
                "Notes",
            ]
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df["TargetDate"] = pd.to_datetime(df["TargetDate"])

    # în caz că, vreodată, vom avea mai multe runde de forecast pentru aceeași săptămână
    df = df.sort_values("Date")
    df = (
        df.groupby(["WeekStart", "TargetDate", "Ticker", "ModelVersion"])
        .tail(1)
        .reset_index(drop=True)
    )
    return df


def compute_realized_returns(
    forecasts: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Pentru fiecare rând din forecasts:
      - luăm prețul de referință la începutul săptămânii (ultimul preț <= WeekStart)
      - luăm prețul la finalul săptămânii (ultimul preț <= TargetDate)
      - calculăm randamentul realizat în %
    """

    if forecasts.empty:
        return pd.DataFrame()

    if prices.empty:
        print("[EVAL] No prices, cannot compute realized returns.")
        return pd.DataFrame()

    today = date.today()
    today_ts = pd.Timestamp(today)

    # Evaluăm doar săptămânile care s-au terminat deja
    eligible = forecasts[forecasts["TargetDate"] < today_ts].copy()
    if eligible.empty:
        print("[EVAL] No completed weeks to evaluate yet.")
        return pd.DataFrame()

    prices = prices.sort_values("Date")

    # pregătim subsetul de coloane pentru merge_asof
    price_cols = ["Date", "Ticker", "Close"]
    p = prices[price_cols].sort_values("Date")

    # sortăm și forecasts pentru merge_asof
    eligible = eligible.sort_values("WeekStart")

    # 1) preț la începutul săptămânii: ultimul Close <= WeekStart
    start_merge = pd.merge_asof(
        eligible,
        p,
        left_on="WeekStart",
        right_on="Date",
        by="Ticker",
        direction="backward",
        suffixes=("", "_start"),
    )
    start_merge = start_merge.rename(
        columns={"Date_start": "PriceDateStart", "Close": "CloseStart"}
    )

    # 2) preț la finalul săptămânii: ultimul Close <= TargetDate
    start_merge = start_merge.sort_values("TargetDate")
    end_merge = pd.merge_asof(
        start_merge,
        p,
        left_on="TargetDate",
        right_on="Date",
        by="Ticker",
        direction="backward",
        suffixes=("", "_end"),
    )
    end_merge = end_merge.rename(
        columns={"Date_end": "PriceDateEnd", "Close": "CloseEnd"}
    )

    df = end_merge

    # calculăm randamentul realizat
    df["Realized_Pct"] = (df["CloseEnd"] / df["CloseStart"] - 1.0) * 100.0

    # eroare de forecast
    df["Error_Pct"] = df["Realized_Pct"] - df["ER_Pct"]
    df["AbsError_Pct"] = df["Error_Pct"].abs()
    df["HitDirection"] = (df["Realized_Pct"] * df["ER_Pct"] > 0).astype(int)

    # coloane ordonate frumos
    ordered_cols = [
        "Date",         # când a fost făcută predicția
        "WeekStart",
        "TargetDate",
        "ModelVersion",
        "Ticker",
        "Region",
        "HorizonDays",
        "ER_Pct",
        "CloseStart",
        "CloseEnd",
        "Realized_Pct",
        "Error_Pct",
        "AbsError_Pct",
        "HitDirection",
        "Notes",
    ]

    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[ordered_cols].copy()


def save_weekly_eval(results: pd.DataFrame) -> None:
    if results.empty:
        print("[EVAL] Nothing to save.")
        return

    reports_dir = settings.REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)
    path = reports_dir / "weekly_eval.csv"

    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, results], ignore_index=True)
        combined.drop_duplicates(
            subset=["WeekStart", "TargetDate", "Ticker", "ModelVersion"],
            keep="last",
            inplace=True,
        )
        combined.to_csv(path, index=False)
    else:
        results.to_csv(path, index=False)

    print(f"[EVAL] Saved {len(results)} rows into {path}")


def main():
    print("[EVAL] Loading prices_history...")
    prices = load_prices_history()

    print("[EVAL] Loading forecasts_stockd...")
    forecasts = load_forecasts()

    print("[EVAL] Computing realized returns...")
    results = compute_realized_returns(forecasts, prices)

    save_weekly_eval(results)


if __name__ == "__main__":
    main()
