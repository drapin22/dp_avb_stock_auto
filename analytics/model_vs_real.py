import pandas as pd
from datetime import timedelta
from pathlib import Path

from stockd import settings


def load_prices() -> pd.DataFrame:
    path = settings.PRICES_HISTORY
    if not path.exists():
        raise FileNotFoundError(f"No prices_history.csv at {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_forecasts(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = settings.DATA_DIR / "forecasts_stockd.csv"
    if not path.exists():
        raise FileNotFoundError(f"No forecasts_stockd.csv at {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def evaluate_forecasts(
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    max_horizon_days: int = 5,
) -> pd.DataFrame:
    """
    Conectează fiecare forecast de tip:
      (Date=T0, Ticker, Region, HorizonDays, ER_Pct)
    cu prețul realizat la T0 + HorizonDays.
    """

    # index ușor pentru lookup: (Ticker, Date) -> Close
    price_index = (
        prices.set_index(["Ticker", "Date"])[["Close", "Region", "Currency"]]
    )

    records = []

    for _, row in forecasts.iterrows():
        t0 = row["Date"]
        ticker = row["Ticker"]
        region = row["Region"]
        horizon = int(row["HorizonDays"])
        model_er = float(row["ER_Pct"]) / 100.0  # transformăm în fracție

        if horizon > max_horizon_days:
            continue

        t_h = t0 + timedelta(days=horizon)

        # preț T0
        try:
            p0 = float(price_index.loc[(ticker, t0), "Close"])
        except KeyError:
            # nu avem preț de start -> forecast imposibil de testat
            continue

        # preț la orizont; dacă nu avem exact t_h, căutăm cel mai apropiat >= t_h
        try:
            # toate datele pentru ticker
            ticker_prices = prices[prices["Ticker"] == ticker].sort_values("Date")
            future = ticker_prices[ticker_prices["Date"] >= t_h]
            if future.empty:
                continue
            ph = float(future.iloc[0]["Close"])
            t_effective = future.iloc[0]["Date"]
        except Exception:
            continue

        realized_ret = (ph / p0) - 1.0

        # direcție: semn
        model_dir = 1 if model_er > 0 else (-1 if model_er < 0 else 0)
        real_dir = 1 if realized_ret > 0 else (-1 if realized_ret < 0 else 0)
        dir_hit = model_dir == real_dir if model_dir != 0 else False

        amplitude_error = realized_ret - model_er
        abs_error = abs(amplitude_error)

        records.append(
            {
                "ModelDate": t0.date(),
                "EffectiveDate": t_effective.date(),
                "Ticker": ticker,
                "Region": region,
                "HorizonDays": horizon,
                "ModelVersion": row.get("ModelVersion", ""),
                "Model_ER_Pct": model_er * 100,
                "Realized_Pct": realized_ret * 100,
                "AmplitudeError_Pct": amplitude_error * 100,
                "AbsError_Pct": abs_error * 100,
                "DirectionHit": dir_hit,
            }
        )

    return pd.DataFrame(records)


def summarize_stats(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce un rezumat pe:
      - Region
      - HorizonDays
      - ModelVersion
    cu hit rate, eroare medie, bias etc.
    """
    if eval_df.empty:
        return pd.DataFrame()

    g = eval_df.groupby(["ModelVersion", "Region", "HorizonDays"])

    summary = g.agg(
        n_forecasts=("Ticker", "count"),
        hit_rate=("DirectionHit", "mean"),
        mean_error_pct=("AmplitudeError_Pct", "mean"),
        median_error_pct=("AmplitudeError_Pct", "median"),
        mean_abs_error_pct=("AbsError_Pct", "mean"),
    ).reset_index()

    # convertim în procente (0.65 -> 65%)
    summary["hit_rate"] = summary["hit_rate"] * 100.0

    return summary


def main():
    prices = load_prices()
    forecasts = load_forecasts()

    eval_df = evaluate_forecasts(prices, forecasts, max_horizon_days=5)
    if eval_df.empty:
        print("[EVAL] No evaluable forecasts.")
        return

    summary = summarize_stats(eval_df)

    # salvăm și în CSV, nu doar în log
    out_dir = settings.DATA_DIR
    eval_path = out_dir / "model_eval_detailed.csv"
    summary_path = out_dir / "model_eval_summary.csv"

    eval_df.to_csv(eval_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[EVAL] Saved detailed results to", eval_path)
    print("[EVAL] Saved summary stats to", summary_path)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
