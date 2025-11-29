# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
import pandas as pd
from openai import OpenAI

from stockd import settings

# Cheia e citită automat din variabila de mediu OPENAI_API_KEY
client = OpenAI()


def _build_prompt(holdings: pd.DataFrame,
                  prices_history: pd.DataFrame,
                  as_of: date,
                  horizon_days: int) -> str:
    """
    Construiește promptul trimis către model.
    Îi dăm:
      - lista de dețineri
      - ultimele ~60 de zile de prețuri pentru acele tickere
    și îi cerem STRICT un JSON cu forecasts.
    """

    holdings_records = holdings.to_dict(orient="records")

    tickers = sorted(set(holdings["Ticker"]))
    recent = prices_history[prices_history["Ticker"].isin(tickers)].copy()

    if not recent.empty:
        cutoff = recent["Date"].max() - pd.Timedelta(days=60)
        recent = recent[recent["Date"] >= cutoff]
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    price_records = recent.to_dict(orient="records")

    prompt = f"""
You are STOCKD, a systematic forecasting model for stocks.

Today is {as_of.isoformat()}.
You must forecast the expected percentage return (ER_Pct) for each ticker
for the NEXT {horizon_days} calendar days (typically a 5-day trading week).

INPUT DATA
----------
Holdings (current portfolio positions):
{json.dumps(holdings_records, indent=2)}

Recent daily prices for these tickers (last ~60 days):
{json.dumps(price_records, indent=2)}

REQUIREMENTS
------------
1. Think using your own internal reasoning, but DO NOT include that reasoning
   in the final answer.
2. The ONLY thing you are allowed to output is a single JSON object
   that matches exactly this schema:

{{
  "forecasts": [
    {{
      "Ticker": "STRING (one of the tickers from Holdings)",
      "Region": "STRING (RO / EU / US)",
      "ER_Pct": NUMBER   // expected total return over the horizon, in percent
    }},
    ...
  ]
}}

3. Include ALL tickers from Holdings in the forecasts list.
4. Do NOT wrap the JSON in markdown.
5. Do NOT add any extra keys, comments, or text outside the JSON.
6. ER_Pct can be negative or positive (e.g. -3.5, 0.0, 4.2).

Return ONLY that JSON object.
"""
    return prompt.strip()


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Rulează modelul STOCKD (bazat pe OpenAI) și întoarce un DataFrame cu:
      ['Ticker', 'Region', 'ER_Pct'].

    Dacă ceva eșuează la apelul OpenAI sau la parsarea JSON,
    întoarce fallback cu ER_Pct = 0.0 pentru toate tickerele.
    """

    if holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    prompt = _build_prompt(holdings, prices_history, as_of, horizon_days)

    try:
        # Apelăm endpointul clasic de chat, fără response_format,
        # ca să fim compatibili cu orice versiune de librărie.
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1200,
        )

        # Extragem textul brut trimis de model
        content = response.choices[0].message.content
        # Încercăm să parsăm direct ca JSON
        parsed = json.loads(content)
        forecasts = parsed.get("forecasts", [])

        if not isinstance(forecasts, list):
            raise ValueError("`forecasts` is not a list in model output")

    except Exception as exc:
        print(f"[STOCKD] ERROR calling/parsing OpenAI response: {exc}")
        # Fallback sigur: 0% ER pentru toate tickerele
        fb = holdings.copy()
        fb["ER_Pct"] = 0.0
        return fb[["Ticker", "Region", "ER_Pct"]]

    # Construim DataFrame din JSON
    df = pd.DataFrame(forecasts)

    # Ne asigurăm că avem coloanele necesare
    for col in ["Ticker", "Region", "ER_Pct"]:
        if col not in df.columns:
            df[col] = None

    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    # Facem merge cu holdings ca să nu pierdem nimic și să avem exact
    # aceiași tickeri / regiuni ca în portofoliu
    merged = holdings.merge(
        df[["Ticker", "Region", "ER_Pct"]],
        on=["Ticker", "Region"],
        how="left",
    )

    merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)

    return merged[["Ticker", "Region", "ER_Pct"]]
