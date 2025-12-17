# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()


def _compute_ticker_features(
    prices_history: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature-uri per (Ticker, Region):
      - last_close
      - ret_5d, ret_20d, ret_60d (%)
      - vol_20d (%)
      - max_dd_60d (%)
    """
    if prices_history.empty or holdings.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Region", "last_close",
            "ret_5d", "ret_20d", "ret_60d",
            "vol_20d", "max_dd_60d",
        ])

    needed_cols = {"Date", "Ticker", "Region", "Close"}
    if not needed_cols.issubset(set(prices_history.columns)):
        raise ValueError(f"prices_history missing columns: {needed_cols - set(prices_history.columns)}")

    ph = prices_history.copy()
    ph["Date"] = pd.to_datetime(ph["Date"])
    ph["Close"] = pd.to_numeric(ph["Close"], errors="coerce")
    ph = ph.dropna(subset=["Close"])

    hold = holdings[["Ticker", "Region"]].drop_duplicates().copy()
    ph = ph.merge(hold, on=["Ticker", "Region"], how="inner")
    if ph.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Region", "last_close",
            "ret_5d", "ret_20d", "ret_60d",
            "vol_20d", "max_dd_60d",
        ])

    cutoff = ph["Date"].max() - pd.Timedelta(days=80)
    ph = ph[ph["Date"] >= cutoff].sort_values(["Ticker", "Region", "Date"]).copy()

    out = []
    for (tkr, reg), g in ph.groupby(["Ticker", "Region"], sort=False):
        g = g.sort_values("Date").copy()
        closes = g["Close"].astype(float).values

        last_close = float(closes[-1])

        def pct_return(n: int) -> float:
            if len(closes) < n + 1:
                return float("nan")
            return float((closes[-1] / closes[-(n + 1)] - 1.0) * 100.0)

        ret_5d = pct_return(5)
        ret_20d = pct_return(20)
        ret_60d = pct_return(60)

        if len(closes) >= 21:
            daily_ret = pd.Series(closes).pct_change()
            vol_20d = float(daily_ret.tail(20).std() * (20 ** 0.5) * 100.0)
        else:
            vol_20d = float("nan")

        g_tail = g.tail(60).copy()
        roll_max = g_tail["Close"].cummax()
        dd = g_tail["Close"] / roll_max - 1.0
        max_dd_60d = float(dd.min() * 100.0) if not dd.empty else float("nan")

        out.append({
            "Ticker": tkr,
            "Region": reg,
            "last_close": last_close,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "ret_60d": ret_60d,
            "vol_20d": vol_20d,
            "max_dd_60d": max_dd_60d,
        })

    return pd.DataFrame(out)


def _compute_region_summary(ticker_features: pd.DataFrame) -> pd.DataFrame:
    if ticker_features.empty:
        return pd.DataFrame(columns=[
            "Region", "avg_ret_5d", "avg_ret_20d", "avg_ret_60d",
            "avg_vol_20d", "avg_max_dd_60d", "risk_regime",
        ])

    agg = (
        ticker_features
        .groupby("Region")[["ret_5d", "ret_20d", "ret_60d", "vol_20d", "max_dd_60d"]]
        .mean()
        .reset_index()
        .rename(columns={
            "ret_5d": "avg_ret_5d",
            "ret_20d": "avg_ret_20d",
            "ret_60d": "avg_ret_60d",
            "vol_20d": "avg_vol_20d",
            "max_dd_60d": "avg_max_dd_60d",
        })
    )

    regimes = []
    for _, row in agg.iterrows():
        m20 = row["avg_ret_20d"]
        vol = row["avg_vol_20d"]
        dd = row["avg_max_dd_60d"]
        if pd.isna(m20):
            regime = "UNKNOWN"
        elif m20 > 3 and dd > -10:
            regime = "RISK_ON"
        elif m20 < -3 and dd < -15:
            regime = "RISK_OFF"
        elif (not pd.isna(vol)) and vol > 8 and dd < -12:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"
        regimes.append(regime)

    agg["risk_regime"] = regimes
    return agg


def _build_prompt(
    holdings: pd.DataFrame,
    ticker_features: pd.DataFrame,
    region_summary: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> str:
    hold_records = holdings[["Ticker", "Region"]].to_dict(orient="records")
    feat_records = ticker_features.to_dict(orient="records")
    reg_records = region_summary.to_dict(orient="records")

    prompt = f"""
You are {settings.MODEL_VERSION_TAG}, a systematic multi-region equity forecaster.

You DO NOT have live news feeds. You approximate regimes using price features:
  - momentum (5/20/60 days),
  - volatility (20d),
  - max drawdown (60d),
  - region-level averages.

Date: {as_of.isoformat()}
Forecast horizon: next {horizon_days} calendar days (about one trading week).

Return a JSON object with "forecasts": list of entries:
  - "Ticker" (string, exactly as input)
  - "Region" (RO/EU/US)
  - "ER_Pct" (number, expected % return over horizon; can be negative)

Guidelines:
  - Conservative magnitudes; typical weekly ER rarely exceeds +/-10%.
  - If signal is weak, keep ER near 0.
  - Use cross-sectional context within each Region.

HOLDINGS:
{json.dumps(hold_records, indent=2)}

TICKER FEATURES:
{json.dumps(feat_records, indent=2)}

REGION SUMMARY:
{json.dumps(reg_records, indent=2)}
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

    holdings = holdings[["Ticker", "Region"]].drop_duplicates().copy()

    ticker_features = _compute_ticker_features(prices_history, holdings)
    region_summary = _compute_region_summary(ticker_features)

    prompt = _build_prompt(
        holdings=holdings,
        ticker_features=ticker_features,
        region_summary=region_summary,
        as_of=as_of,
        horizon_days=horizon_days,
    )

    schema: Dict[str, Any] = {
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
                            "ER_Pct": {"type": "number"},
                        },
                        "required": ["Ticker", "Region", "ER_Pct"],
                    },
                }
            },
            "required": ["forecasts"],
            "additionalProperties": False,
        },
    }

    # Retry/backoff pentru stabilitate
    last_exc = None
    for attempt in range(1, 4):
        try:
            response = client.responses.create(
                model=settings.MODEL_NAME,
                input=prompt,
                response_format={"type": "json_schema", "json_schema": schema},
                max_output_tokens=1200,
            )
            raw_json = response.output[0].content[0].text
            parsed = json.loads(raw_json)
            forecasts = parsed.get("forecasts", [])
            df = pd.DataFrame(forecasts)
            if df.empty:
                raise ValueError("Empty forecasts from model.")
            df["ER_Pct"] = pd.to_numeric(df.get("ER_Pct", 0.0), errors="coerce").fillna(0.0)
            df = df[["Ticker", "Region", "ER_Pct"]].copy()
            merged = holdings.merge(df, on=["Ticker", "Region"], how="left")
            merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)
            return merged
        except Exception as exc:
            last_exc = exc
            sleep_s = 2 ** attempt
            print(f"[STOCKD] OpenAI call failed (attempt {attempt}/3): {exc}. Retrying in {sleep_s}s")
            time.sleep(sleep_s)

    print(f"[STOCKD] ERROR calling OpenAI after retries: {last_exc}")
    fb = holdings.copy()
    fb["ER_Pct"] = 0.0
    return fb
