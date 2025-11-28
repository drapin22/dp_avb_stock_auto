import pandas as pd
from datetime import date
from stockd import settings


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_forecasts() -> pd.DataFrame:
    path = settings.DATA_DIR / "forecasts_stockd.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df["TargetDate"] = pd.to_datetime(df["TargetDate"])
    return df


def evaluate_forecasts(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Pentru fiecare forecast:
      - p0 = preț în ziua Date
      - pT = primul preț disponibil la sau după TargetDate
      - Realized_Pct = (pT / p0 - 1) * 100
    """

    prices = prices.sort_values(["Ticker", "Date"]).copy()

    # index ușor pentru p0
    price_idx = prices.set_index(["Ticker", "Date"])["Close"]

    records = []

    today = date.today()

    for _, f in forecasts.iterrows():
        ticker = f["Ticker"]
        t0 = f["Date"]
        target = f["TargetDate"]
        week_start = f["WeekStart"].date()
        model_version = f.get("ModelVersion", "")
        er_pct = float(f["ER_Pct"])

        # evaluăm DOAR dacă TargetDate <= azi (săptămână „închisă” sau în curs)
        if target.date() > today:
            continue

        # prețul inițial p0
        try:
            p0 = float(price_idx.loc[(ticker, t0)])
        except KeyError:
            # dacă nu avem exact t0, luăm primul >= t0
            tp = prices[prices["Ticker"] == ticker]
            tp = tp[tp["Date"] >= t0]
            if tp.empty:
                continue
            p0 = float(tp.iloc[0]["Close"])
            t0 = tp.iloc[0]["Date"]

        # prețul la sau după TargetDate
        tp = prices[prices["Ticker"] == ticker]
        tp = tp[tp["Date"] >= target]
        if tp.empty:
            # nu avem încă prețurile până la target, nu evaluăm
            continue
        pT = float(tp.iloc[0]["Close"])
        t_effective = tp.iloc[0]["Date"]

        realized_pct = (pT / p0 - 1.0) * 100.0

        model_dir = 1 if er_pct > 0 else (-1 if er_pct < 0 else 0)
        real_dir = 1 if realized_pct > 0 else (-1 if realized_pct < 0 else 0)
        direction_hit = model_dir == real_dir if model_dir != 0 else False

        error_pct = realized_pct - er_pct
        abs_error_pct = abs(error_pct)

        records.append(
            {
                "WeekStart": week_start,
                "ModelDate": t0.date(),
                "TargetDate": target.date(),
                "EffectiveDate": t_effective.date(),
                "ModelVersion": model_version,
                "Ticker": ticker,
                "Region": f["Region"],
                "HorizonDays": int(f["HorizonDays"]),
                "Model_ER_Pct": er_pct,
                "Realized_Pct": realized_pct,
                "Error_Pct": error_pct,
                "AbsError_Pct": abs_error_pct,
                "DirectionHit": direction_hit,
            }
        )

    return pd.DataFrame(records)


def summarize_weekly(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rezumat pe:
      - WeekStart
      - ModelVersion
      - Region (RO / EU / US)
    """

    if eval_df.empty:
        return pd.DataFrame()

    g = eval_df.groupby(["WeekStart", "ModelVersion", "Region"])

    summary = g.agg(
        n_forecasts=("Ticker", "count"),
        hit_rate=("DirectionHit", "mean"),
        mean_error_pct=("Error_Pct", "mean"),
        median_error_pct=("Error_Pct", "median"),
        mean_abs_error_pct=("AbsError_Pct", "mean"),
    ).reset_index()

    summary["hit_rate"] = summary["hit_rate"] * 100.0

    return summary.sort_values(["WeekStart", "Region"])


def main():
    prices = load_prices()
    forecasts = load_forecasts()

    eval_df = evaluate_forecasts(prices, forecasts)
    if eval_df.empty:
        print("[EVAL] No evaluable forecasts yet.")
        return

    summary = summarize_weekly(eval_df)

    # salvăm totul
    eval_path = settings.DATA_DIR / "model_eval_detailed.csv"
    summary_path = settings.DATA_DIR / "model_eval_summary.csv"

    eval_df.to_csv(eval_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[EVAL] Saved detailed results to", eval_path)
    print("[EVAL] Saved weekly summary to", summary_path)
    print("\n=== WEEKLY SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
