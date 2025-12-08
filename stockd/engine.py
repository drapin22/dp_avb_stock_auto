# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
from typing import Dict, Any

import numpy as pd
import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()  # ia cheia din environment (GitHub secret OPENAI_API_KEY)


# --------- HELPERI PENTRU FEATURE-URI DE PREȚ -----------------


def _compute_ticker_features(
    prices_history: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Construiește feature-uri agregate pentru fiecare ticker din holdings:
      - last_close
      - ret_5d, ret_20d, ret_60d (%)
      - vol_20d (%)
      - max_dd_60d (% drawdown max din ultimele 60 zile)
    """

    if prices_history.empty or holdings.empty:
        return pd.DataFrame(columns=[
            "Ticker",
            "Region",
            "last_close",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "vol_20d",
            "max_dd_60d",
        ])

    tickers = holdings["Ticker"].unique().tolist()
    df = prices_history.copy()
    df = df[df["Ticker"].isin(tickers)].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "Ticker",
            "Region",
            "last_close",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "vol_20d",
            "max_dd_60d",
        ])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Păstrăm ultimele 70 de zile doar ca să nu explodeze prompt-ul
    cutoff = df["Date"].max() - pd.Timedelta(days=70)
    df = df[df["Date"] >= cutoff].copy()

    features = []

    for (ticker, region), g in df.groupby(["Ticker", "Region"]):
        g = g.sort_values("Date")
        g["Close"] = g["Close"].astype(float)

        last_close = float(g["Close"].iloc[-1])

        def pct_return(n: int) -> float:
            if len(g) < n + 1:
                return float("nan")
            start = g["Close"].iloc[-(n + 1)]
            end = g["Close"].iloc[-1]
            return float((end / start - 1.0) * 100.0)

        ret_5d = pct_return(5)
        ret_20d = pct_return(20)
        ret_60d = pct_return(60)

        # Volatilitate realizată pe 20 zile (std daily * sqrt(20))
        if len(g) >= 20:
            daily_ret = g["Close"].pct_change()
            vol_20d = float(daily_ret.tail(20).std() * (20 ** 0.5) * 100.0)
        else:
            vol_20d = float("nan")

        # Max drawdown pe 60 zile
        g_tail = g.tail(60).copy()
        roll_max = g_tail["Close"].cummax()
        drawdown = g_tail["Close"] / roll_max - 1.0
        max_dd_60d = float(drawdown.min() * 100.0) if not drawdown.empty else float("nan")

        features.append(
            {
                "Ticker": ticker,
                "Region": region,
                "last_close": last_close,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "vol_20d": vol_20d,
                "max_dd_60d": max_dd_60d,
            }
        )

    return pd.DataFrame(features)


def _compute_region_summary(ticker_features: pd.DataFrame) -> pd.DataFrame:
    """
    Sumar pe regiune: medii ale feature-urilor.
    Folosit ca proxy pentru:
      - regim de risk-on / risk-off
      - ciclu local vs global
    """

    if ticker_features.empty:
        return pd.DataFrame(columns=[
            "Region",
            "avg_ret_5d",
            "avg_ret_20d",
            "avg_ret_60d",
            "avg_vol_20d",
            "avg_max_dd_60d",
            "risk_regime",
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

    # Etichetă grosieră de regim
    regimes = []
    for _, row in agg.iterrows():
        m20 = row["avg_ret_20d"]
        vol = row["avg_vol_20d"]
        dd = row["avg_max_dd_60d"]  # negativ

        if pd.isna(m20):
            regime = "UNKNOWN"
        elif m20 > 3 and dd > -10:
            regime = "RISK_ON"
        elif m20 < -3 and dd < -15:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        regimes.append(regime)

    agg["risk_regime"] = regimes
    return agg


# --------- BUILD PROMPT -----------------


def _build_prompt(
    holdings: pd.DataFrame,
    ticker_features: pd.DataFrame,
    region_summary: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> str:
    holdings_records = holdings.to_dict(orient="records")
    feat_records = ticker_features.to_dict(orient="records")
    region_records = region_summary.to_dict(orient="records")

    prompt = f"""
You are STOCKD_V10.7F+, a systematic, multi-region equity forecaster.

You DO NOT have live news feeds or macro databases.
Instead, you approximate cycles, sentiment and risk regimes using:
  - multi-horizon price momentum (5, 20, 60 days),
  - realized volatility,
  - max drawdown,
  - region-level averages of these features.

Today's date: {as_of.isoformat()}.
Forecast horizon: next {horizon_days} calendar days (roughly 1 trading week).

For each ticker in the portfolio, you must estimate the *expected percentage price return*
over this horizon (ER_Pct), taking into account:
  - its own trend and drawdown vs history,
  - its volatility,
  - the risk regime of its Region (RO, EU, US),
  - cross-section: how strong/weak it is vs other names in same Region.

Assume:
  - RO tickers are Romanian BVB equities (smaller, less liquid, more idiosyncratic).
  - EU tickers are Western European large/mid caps.
  - US tickers are US-listed names with higher liquidity.
Macro and news effects are reflected indirectly through prices and regimes.

Guidelines:
  - Use conservative magnitudes. Weekly ER_Pct rarely exceeds +/-10% for normal names.
  - If signal is weak or regime is very uncertain, keep ER_Pct close to 0.
  - Keep RO more volatile than EU, and EU slightly less volatile than US growth names.

You must return a JSON object with a list "forecasts".
Each forecast entry must have:
  - "Ticker": string (exactly as given in the input),
  - "Region": string ("RO", "EU" or "US"),
  - "ER_Pct": number (expected price return % over the horizon, can be negative).

PORTFOLIO HOLDINGS (current positions):
{json.dumps(holdings_records, indent=2)}

TICKER FEATURES (per name, based on last ~70 days):
- last_close: last closing price in local currency
- ret_5d/20d/60d: % price return over that window
- vol_20d: realized volatility over last 20 days (annualized in %, approx.)
- max_dd_60d: worst drawdown over last 60 days in %

{json.dumps(feat_records, indent=2)}

REGION SUMMARY (cycle / risk regime proxies):
- average returns and volatility across names in the region
- simple risk_regime flag: RISK_ON / RISK_OFF / NEUTRAL / UNKNOWN

{json.dumps(region_records, indent=2)}
"""
    return prompt.strip()


# --------- ENTRYPOINT: MODEL INTERFACE -----------------


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Interfața oficială a modelului tău StockD.

    Input:
      - holdings: DataFrame cu cel puțin ['Ticker', 'Region']
      - prices_history: toate prețurile istorice (RO + EU + US)
      - as_of: data la care rulezi modelul
      - horizon_days: orizontul în zile pentru care vrei ER

    Output:
      - DataFrame cu:
          ['Ticker', 'Region', 'ER_Pct']
    """

    if holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct"])

    # 1) Feature-uri per ticker
    ticker_features = _compute_ticker_features(prices_history, holdings, as_of)

    # 2) Sumar pe regiune (proxy ciclo / sentiment)
    region_summary = _compute_region_summary(ticker_features)

    # 3) Prompt
    prompt = _build_prompt(
        holdings=holdings,
        ticker_features=ticker_features,
        region_summary=region_summary,
        as_of=as_of,
        horizon_days=horizon_days,
    )

    # 4) Schema pentru răspuns JSON
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

    try:
        response = client.responses.create(
            model=settings.MODEL_NAME,
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
            max_output_tokens=1200,
        )

        raw_json = response.output[0].content[0].text
        parsed = json.loads(raw_json)
        forecasts = parsed.get("forecasts", [])

    except Exception as exc:
        print(f"[STOCKD] ERROR calling OpenAI: {exc}")
        fb = holdings.copy()
        fb["ER_Pct"] = 0.0
        return fb[["Ticker", "Region", "ER_Pct"]]

    if not forecasts:
        fb = holdings.copy()
        fb["ER_Pct"] = 0.0
        return fb[["Ticker", "Region", "ER_Pct"]]

    df = pd.DataFrame(forecasts)

    # Curățare minimă
    if "ER_Pct" not in df.columns:
        df["ER_Pct"] = 0.0

    df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

    # Asigurăm acoperire pentru toate holding-urile
    merged = holdings.merge(df, on=["Ticker", "Region"], how="left")
    merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)

    return merged[["Ticker", "Region", "ER_Pct"]]
