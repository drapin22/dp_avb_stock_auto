# stockd/model_weekly.py
from __future__ import annotations

from datetime import date, timedelta
import os
import pandas as pd

from stockd import settings
from stockd.engine import run_stockd_model
from stockd.notify import send_telegram_message


def get_next_monday(d: date) -> date:
    """Returnează următoarea zi de luni după data d."""
    # Monday = 0, Sunday = 6
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return d + timedelta(days=days_until_monday)


def _debug_print_holdings(df: pd.DataFrame, label: str) -> None:
    print(f"[DEBUG] {label}: {len(df)} rows RAW")
    if "Active" in df.columns:
        after = df[df["Active"] == 1]
        print(
            f"[DEBUG] {label}: {len(df)} -> {len(after)} rows after filter Active=1"
        )


def load_all_holdings() -> pd.DataFrame:
    """
    Încarcă toate deținerile din fișierele:
      - holdings_ro.csv
      - holdings_eu.csv
      - holdings_us.csv

    Returnează un DataFrame cu minim coloanele: ['Ticker', 'Region'].
    """
    sources = [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]

    print("[MODEL] Loading holdings...")
    dfs: list[pd.DataFrame] = []

    for path, region_default in sources:
        print(f"[DEBUG] Loading holdings from: {path} (default Region={region_default})")

        if not path.exists():
            print(f"[DEBUG]   File not found, skipping.")
            continue

        df = pd.read_csv(path)

        _debug_print_holdings(df, path.name)

        # Filtru Active=1 dacă există coloana
        if "Active" in df.columns:
            df = df[df["Active"] == 1]

        # Asigurăm Region
        if "Region" not in df.columns:
            df["Region"] = region_default

        # Ne asigurăm că avem coloanele de care avem nevoie
        if "Ticker" not in df.columns:
            raise ValueError(f"{path} must contain a 'Ticker' column.")

        dfs.append(df[["Ticker", "Region"]])

    if not dfs:
        print("[MODEL] No holdings found in any region.")
        return pd.DataFrame(columns=["Ticker", "Region"])

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"[DEBUG] Combined holdings: {len(combined)} rows")
    return combined


def load_prices_history() -> pd.DataFrame:
    """
    Încarcă istoricul de prețuri din prices_history.csv.
    """
    if not settings.PRICES_HISTORY.exists():
        print("[DEBUG] prices_history.csv not found, using empty DataFrame.")
        return pd.DataFrame(
            columns=["Date", "Ticker", "Region", "Currency", "Close"]
        )

    df = pd.read_csv(settings.PRICES_HISTORY)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    print(f"[DEBUG] Loaded prices_history.csv: {len(df)} rows")
    return df


def run_stockd_weekly_model() -> None:
    """
    Rulează modelul StockD pentru săptămâna următoare:

      - stabilește week_start (următoarea luni) și target_date (vineri)
      - încarcă holdings + prices_history
      - cheamă run_stockd_model (OpenAI)
      - salvează predicțiile în data/forecasts_stockd.csv
      - trimite un rezumat pe Telegram cu TOATE tickerele:
          Ticker, Region, Preț curent, ER%, Preț țintă
    """
    today = date.today()
    week_start = get_next_monday(today)
    target_date = week_start + timedelta(days=4)  # luni → vineri
    horizon_days = (target_date - week_start).days + 1  # 5 zile

    print(
        f"[MODEL] Today = {today}, forecasting week {week_start} – {target_date}"
    )

    # 1. Holdings
    holdings = load_all_holdings()
    if holdings.empty:
        print("[MODEL] No holdings found. Nothing to forecast.")
        return

    # 2. Prețuri istorice
    prices_history = load_prices_history()

    # 3. Chemăm modelul StockD (OpenAI)
    preds = run_stockd_model(
        holdings=holdings,
        prices_history=prices_history,
        as_of=today,
        horizon_days=horizon_days,
    )

    if "ER_Pct" not in preds.columns:
        raise ValueError("run_stockd_model must return a column named 'ER_Pct'.")

    # Ne asigurăm că avem 1 rând / Ticker / Region
    merged = holdings.merge(
        preds[["Ticker", "Region", "ER_Pct"]],
        on=["Ticker", "Region"],
        how="left",
    )

    if merged["ER_Pct"].isna().any():
        missing = merged[merged["ER_Pct"].isna()][["Ticker", "Region"]]
        print("[MODEL] WARNING: missing ER_Pct for some tickers:")
        print(missing.to_string(index=False))
        merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)

    # 4. Construim DataFrame-ul pentru forecasts_stockd.csv
    rows: list[dict] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Date": today.strftime("%Y-%m-%d"),          # când a rulat modelul
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

    # 5. Salvăm în data/forecasts_stockd.csv (cu dedup)
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

    # === 6. Pregătim datele pentru mesajul de Telegram ===
    try:
        # Ultimul preț pentru fiecare (Ticker, Region)
        if prices_history.empty:
            latest = pd.DataFrame(
                columns=["Ticker", "Region", "Close"]
            ).set_index(["Ticker", "Region"])
        else:
            latest = (
                prices_history.sort_values("Date")
                .groupby(["Ticker", "Region"])
                .tail(1)
                .set_index(["Ticker", "Region"])
            )

        lines = [
            "📈 StockD weekly forecast",
            f"Săptămână: {week_start} → {target_date}",
            "",
            "Ticker (Reg) | Price → Target | ER%",
        ]

        # sortăm frumos pe regiune și ticker
        for _, r in new_df.sort_values(["Region", "Ticker"]).iterrows():
            key = (r["Ticker"], r["Region"])
            er = float(r["ER_Pct"])
            if key in latest.index:
                price = float(latest.loc[key, "Close"])
                target_price = price * (1.0 + er / 100.0)
                lines.append(
                    f"- {r['Ticker']} ({r['Region']}): "
                    f"{price:.2f} → {target_price:.2f} "
                    f"({er:.2f}%)"
                )
            else:
                # dacă nu avem preț, tot vrem să vedem ER-ul
                lines.append(
                    f"- {r['Ticker']} ({r['Region']}): "
                    f"no price, ER {er:.2f}%"
                )

        msg = "\n".join(lines)
        send_telegram_message(msg)
    except Exception as exc:
        print(f"[MODEL] Failed to send Telegram summary: {exc}")


if __name__ == "__main__":
    run_stockd_weekly_model()
