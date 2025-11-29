# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
from typing import List

import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()  # Citește automat OPENAI_API_KEY din env


def _build_prompt(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> str:
    """
    Construim promptul trimis la model.

    Ideea:
      - Îi explicăm clar ce vrem: expected return pe următoarele X zile
      - Îi dăm portofoliul și un mic istoric de prețuri
      - Îi cerem STRICT JSON, fără text în plus
    """

    # 1) serializăm portofoliul
    holdings_records = holdings.to_dict(orient="records")

    # 2) reducem un pic istoria, să nu fie monstru
    #    luăm ultimele 60 de zile pentru tickerele din holdings
    tickers = sorted(set(holdings["Ticker"]))
    recent = prices_history[prices_history["Ticker"].isin(tickers)].copy()
    if not recent.empty:
        cutoff = recent["Date"].max() - pd.Timedelta(days=60)
        recent = recent[recent["Date"] >= cutoff]
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    price_records = recent.sort_values(["Ticker", "Date"]).to_dict(orient="records")

    prompt = f"""
You are STOCKD, a systematic portfolio model.

Today is {as_of.isoformat()}.
You must generate an EXPECTED RETURN forecast for each holding over the NEXT {horizon_days} calendar days
(roughly one trading week, Monday to Friday).

Holdings (each object has Ticker and Region):
{json.dumps(holdings_records, indent=2)}

Recent prices for these tickers (Date, Ticker, Region, Currency, Close):
{json.dumps(price_records, indent=2)}

TASK:
- For every holding in the list, estimate the expected percentage return ER_Pct
  over the next {horizon_days} days.
- You can use trend, volatility, mean reversion, cross-sectional patterns, etc.
- Keep numbers relatively small and realistic (usually between -10 and +10).

OUTPUT:
Return ONLY valid JSON with this exact structure:

{{
  "forecasts": [
    {{"Ticker": "WINE", "Region": "RO", "ER_Pct": 2.5}},
    {{"Ticker": "BABA", "Region": "US", "ER_Pct": -1.2}}
  ]
}}

Rules:
- Include ALL tickers from the holdings list, no extra tickers.
- ER_Pct is a float, not a string.
- Do NOT add any explanation text outside the JSON object.
""".strip()

    return prompt


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Interfața oficială a modelului StockD.

    Returnează DataFrame cu coloanele:
      ['Ticker', 'Region', 'ER_Pct']
    """

    if holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    prompt = _build_prompt(
        holdings=holdings,
        prices_history=prices_history,
        as_of=as_of,
        horizon_days=horizon_days,
    )

    try:
        response = client.responses.create(
            model=settings.MODEL_NAME,
            input=prompt,
            # forțăm JSON valid
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # În caz că pică API-ul: fallback ER=0
        print(f"[STOCKD] ERROR calling OpenAI: {exc}")
        result = holdings[["Ticker", "Region"]].copy()
        result["ER_Pct"] = 0.0
        return result

    # Extragem textul răspunsului
    try:
        # Noua API Responses: textul e în output[0].content[0].text
        raw_text = response.output[0].content[0].text
    except Exception:
        # Dacă structura se schimbă, mai bine log + fallback
        print("[STOCKD] WARNING: unexpected response structure, using fallback ER=0.")
        result = holdings[["Ticker", "Region"]].copy()
        result["ER_Pct"] = 0.0
        return result

    try:
        data = json.loads(raw_text)
        forecasts = data.get("forecasts", [])
    except Exception as exc:
        print(f"[STOCKD] ERROR parsing JSON from model: {exc}")
        forecasts = []

    if not forecasts:
        # fallback: 0 pentru toți, dar păstrăm pipe-ul viu
        print("[STOCKD] WARNING: empty forecasts, defaulting to ER_Pct = 0.")
        result = holdings[["Ticker", "Region"]].copy()
        result["ER_Pct"] = 0.0
        return result

    df = pd.DataFrame(forecasts)

    # Normalizăm și ne asigurăm că avem coloanele corecte
    if "Ticker" not in df.columns or "Region" not in df.columns:
        raise ValueError("Model response must contain 'Ticker' and 'Region' keys.")

    if "ER_Pct" not in df.columns:
        df["ER_Pct"] = 0.0

    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    # ne limităm la tickerele efective din holdings și facem merge
    df = df[["Ticker", "Region", "ER_Pct"]]
    merged = holdings[["Ticker", "Region"]].merge(
        df, on=["Ticker", "Region"], how="left"
    )
    merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)

    return merged
