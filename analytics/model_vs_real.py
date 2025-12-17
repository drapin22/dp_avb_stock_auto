# analytics/model_vs_real.py
import pandas as pd
from datetime import date
from stockd import settings


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(settings.PRICES_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def load_forecasts() -> pd.DataFrame:
    df = pd.read_csv(settings.FORECASTS_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df["TargetDate"] = pd.to_datetime(df["TargetDate"])
    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)
    return df


def evaluate_forecasts(prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["Ticker", "Region", "Date"]).copy()
    today = date.today()

    records = []
    for _, f in forecasts.iterrows():
        ticker = f["Ticker"]
        region = f["Region"]
        t0 = f["Date"]
        target = f["TargetDate"]
        model_version = f.get("ModelVersion", "")
        er_pct = float(f["ER_Pct"])

        if target.date() > today:
            continue

        tp = prices[(prices["Ticker"] == ticker) & (prices["Region"] == region)].copy()
        if tp.empty:
            continue

        # p0: first >= t0
        tp0 = tp[tp["Date"] >= t0]
        if tp0.empty:
            continue
        p0 = float(tp0.iloc[0]["Close"])
        t0_eff = tp0.iloc[0]["Date"]

        # pT: first >= target
        tpT = tp[tp["Date"] >= target]
        if tpT.empty:
            continue
        pT = float(tpT.iloc[0]["Close"])
        t_eff = tpT.iloc[0]["Date"]

        realized_pct = (pT / p0 - 1.0) * 100.0
        model_dir = 1 if er_pct > 0 else (-1 if er_pct < 0 else 0)
        real_dir = 1 if realized_pct > 0 else (-1 if realized_pct < 0 else 0)
        direction_hit = model_dir == real_dir if model_dir != 0 else False

        error_pct = realized_pct - er_pct
        abs_error_pct = abs(error_pct)

        records.append({
            "WeekStart": f["WeekStart"].date(),
            "ModelDate": t0_eff.date(),
            "TargetDate": target.date(),
            "EffectiveDate": t_eff.date(),
            "ModelVersion": model_version,
            "Ticker": ticker,
            "Region": region,
            "HorizonDays": int(f["HorizonDays"]),
            "Model_ER_Pct": er_pct,
            "Realized_Pct": realized_pct,
            "Error_Pct": error_pct,
            "AbsError_Pct": abs_error_pct,
            "DirectionHit": direction_hit,
        })

    return pd.DataFrame(records)


def summarize_weekly(eval_df: pd.DataFrame) -> pd.DataFrame:
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

    eval_path = settings.DATA_DIR / "model_eval_detailed.csv"
    summary_path = settings.DATA_DIR / "model_eval_summary.csv"
    eval_df.to_csv(eval_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
