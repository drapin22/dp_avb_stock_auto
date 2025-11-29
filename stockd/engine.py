# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()   # folosește OPENAI_API_KEY din secrets


def _build_prompt(holdings, prices_history, as_of, horizon_days):
    holdings_records = holdings.to_dict(orient="records")

    # limităm la 60 zile
    tickers = sorted(set(holdings["Ticker"]))
    recent = prices_history[prices_history["Ticker"].isin(tickers)].copy()
    if not recent.empty:
        cutoff = recent["Date"].max() - pd.Timedelta(days=60)
        recent = recent[recent["Date"] >= cutoff]
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    price_records = recent.to_dict(orient="records")

    prompt = f"""
You are STOCKD, a systematic forecast model.
Date today: {as_of.isoformat()}.

Your task:
Predict the expected return (ER_Pct) for EACH ticker over the next {horizon_days} days.
Output MUST be JSON ONLY.

Holdings:
{json.dumps(holdings_records, indent=2)}

Recent prices:
{json.dumps(price_records, indent=2)}

Return ONLY:
{{
  "forecasts": [
    {{"Ticker": "WINE", "Region": "RO", "ER_Pct": 1.2}},
    {{"Ticker": "BABA", "Region": "US", "ER_Pct": -0.4}}
  ]
}}
"""
    return prompt.strip()


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:

    if holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    prompt = _build_prompt(holdings, prices_history, as_of, horizon_days)

    # NOU! — folosim Responses.parse pentru JSON
    try:
        resp = client.responses.parse(
            model=settings.MODEL_NAME,
            input=prompt,
            # îi spunem explicit că vrem JSON
            output_format={"type": "json", "schema": {
                "type": "object",
                "properties": {
                    "forecasts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Ticker": {"type": "string"},
                                "Region": {"type": "string"},
                                "ER_Pct": {"type": "number"}
                            },
                            "required": ["Ticker", "Region", "ER_Pct"]
                        }
                    }
                },
                "required": ["forecasts"]
            }}
        )

        parsed = resp.output_parsed
        forecasts = parsed["forecasts"]

    except Exception as exc:
        # fallback: ER=0
        print(f"[STOCKD] ERROR calling OpenAI: {exc}")
        fallback = holdings.copy()
        fallback["ER_Pct"] = 0.0
        return fallback[["Ticker", "Region", "ER_Pct"]]

    df = pd.DataFrame(forecasts)

    if df.empty:
        fallback = holdings.copy()
        fallback["ER_Pct"] = 0.0
        return fallback

    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    # match exact holdings order
    final = holdings.merge(df, on=["Ticker", "Region"], how="left")
    final["ER_Pct"] = final["ER_Pct"].fillna(0.0)

    return final[["Ticker", "Region", "ER_Pct"]]
