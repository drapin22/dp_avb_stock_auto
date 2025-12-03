import os
from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from stockd.telegram_utils import (
    send_telegram_message,
    send_telegram_document,
    send_telegram_photo,
)

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
    rows = []

    for t in tickers:
        df = prices[prices["Ticker"] == t].sort_values("Date")

        w_open = df[df["Date"] == start_date]
        w_close = df[df["Date"] == end_date]

        if w_open.empty or w_close.empty:
            continue

        open_p = float(w_open["Close"].iloc[0])
        close_p = float(w_close["Close"].iloc[0])
        ret = (close_p - open_p) / open_p * 100.0

        region = df["Region"].iloc[0]

        rows.append(
            {
                "Ticker": t,
                "Region": region,
                "Open": open_p,
                "Close": close_p,
                "Real_Return_Pct": ret,
            }
        )

    return pd.DataFrame(rows)


def generate_plot(df: pd.DataFrame, output_path: str) -> None:
    if df.empty:
        return

    df_sorted = df.sort_values("Error_Pct")

    plt.figure(figsize=(12, 8))
    plt.barh(df_sorted["Ticker"], df_sorted["Error_Pct"])
    plt.title("Eroare predicție vs randament real (pct)")
    plt.xlabel("Eroare (puncte procentuale)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_text_summary(df: pd.DataFrame, week_start, week_end) -> str:
    if df.empty:
        return (
            f"Raport săptămânal StockD {week_start} – {week_end}\n"
            f"Nu am putut calcula randamentele (lipsesc date de preț)."
        )

    df2 = df.copy()
    df2 = df2[["Ticker", "Region", "ER_Pct", "Real_Return_Pct", "Error_Pct"]]

    df2["ER_Pct"] = df2["ER_Pct"].round(2)
    df2["Real_Return_Pct"] = df2["Real_Return_Pct"].round(2)
    df2["Error_Pct"] = df2["Error_Pct"].round(2)

    top = df2.sort_values("Error_Pct").head(5)

    lines = [
        f"Raport săptămânal StockD {week_start} – {week_end}",
        "",
        "Primele 5 poziții după eroare (Real − Predicție, pct):",
    ]

    for _, row in top.iterrows():
        lines.append(
            f"- {row['Ticker']} ({row['Region']}): "
            f"predicție {row['ER_Pct']} pct, "
            f"real {row['Real_Return_Pct']} pct, "
            f"eroare {row['Error_Pct']} pct"
        )

    return "\n".join(lines)


def run_report():
    # calculăm săptămâna anterioară: luni-vineri
    today = date.today()
    last_sunday = today - timedelta(days=today.weekday() + 1)
    week_start = last_sunday + timedelta(days=1)  # luni
    week_end = week_start + timedelta(days=4)     # vineri

    print(f"[REPORT] Week = {week_start} -> {week_end}")

    forecast, prices = load_data()

    # forecast-ul generat duminica pentru această săptămână
    f_week = forecast[forecast["WeekStart"] == str(week_start)]
    if f_week.empty:
        raise ValueError(f"No forecasts found for WeekStart={week_start}")

    returns = compute_weekly_returns(
        prices,
        pd.to_datetime(week_start),
        pd.to_datetime(week_end),
    )

    merged = f_week.merge(returns, on=["Ticker", "Region"], how="left")

    merged["Error_Pct"] = merged["Real_Return_Pct"] - merged["ER_Pct"]

    # fișiere out
    suffix = week_end.strftime("%Y%m%d")
    csv_path = os.path.join(REPORT_DIR, f"weekly_report_{suffix}.csv")
    xlsx_path = os.path.join(REPORT_DIR, f"weekly_report_{suffix}.xlsx")
    png_path = os.path.join(REPORT_DIR, f"weekly_plot_{suffix}.png")

    merged.to_csv(csv_path, index=False)
    merged.to_excel(xlsx_path, index=False)
    generate_plot(merged, png_path)

    print(f"[REPORT] Saved CSV to {csv_path}")
    print(f"[REPORT] Saved XLSX to {xlsx_path}")
    print(f"[REPORT] Saved PNG to {png_path}")

    text_summary = build_text_summary(merged, week_start, week_end)

    return {
        "week_start": week_start,
        "week_end": week_end,
        "summary_text": text_summary,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
        "png_path": png_path,
    }


def run_report_and_notify():
    info = run_report()

    # text
    send_telegram_message(info["summary_text"])

    # atașamente (CSV + Excel + grafic)
    send_telegram_document(
        info["csv_path"],
        caption="Raport săptămânal – date CSV",
    )
    send_telegram_document(
        info["xlsx_path"],
        caption="Raport săptămânal – Excel",
    )
    if os.path.exists(info["png_path"]):
        send_telegram_photo(
            info["png_path"],
            caption="Eroare predicție vs randament real",
        )


if __name__ == "__main__":
    run_report_and_notify()
