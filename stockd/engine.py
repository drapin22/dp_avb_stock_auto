# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()   # ia cheia din secret


def _build_prompt(holdings, prices_history, as_of, horizon_days):
    holdings_records = holdings.to_dict(orient="records")

    tickers = sorted(set(holdings["Ticker"]))
    recent = prices_history[prices_history["Ticker"].isin(tickers)].copy()
    if not recent.empty:
        cutoff = recent["Date"].max() - pd.Timedelta(days=60)
        recent = recent[recent["Date"] >= cutoff]
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    price_records = recent.to_dict(orient="records")

    prompt = f"""
You are STOCKD, a systematic forecasting model.
Date today: {as_of.isoformat()}.

Forecast expected return for each ticker for the next {horizon_days} days.

Return ONLY valid JSON matching the schema.

Holdings:
{json.dumps(holdings_records, indent=2)}

Recent prices (last 60 days):
{json.dumps(price_records, indent=2)}
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

    # JSON Schema VALID pentru Responses
    schema = {
        "name": "stockd_output",
        "schema": {
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
        }
    }

    try:
        response = client.responses.create(
            model=settings.MODEL_NAME,
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            max_output_tokens=1200
        )

        # scoatem JSON-ul
        raw_json = response.output[0].content[0].text
        parsed = json.loads(raw_json)
        forecasts = parsed["forecasts"]

    except Exception as exc:
        print(f"[STOCKD] ERROR calling OpenAI: {exc}")
        # fallback: 0%
        fb = holdings.copy()
        fb["ER_Pct"] = 0.0
        return fb[["Ticker", "Region", "ER_Pct"]]

    df = pd.DataFrame(forecasts)
    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    merged = holdings.merge(df, on=["Ticker", "Region"], how="left")
    merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)

    return merged[["Ticker", "Region", "ER_Pct"]]
