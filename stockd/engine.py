# stockd/engine.py
from __future__ import annotations

from datetime import date
import json
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()


def _compute_ticker_features(prices_history: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    if prices_history.empty or holdings.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Region", "last_close",
            "ret_5d", "ret_20d", "ret_60d",
            "vol_20d", "max_dd_60d",
        ])

    ph = prices_history.copy()
    ph["Date"] = pd.to_datetime(ph["Date"], errors="coerce")
    ph["Close"] = pd.to_numeric(ph["Close"], errors="coerce")
    ph = ph.dropna(subset=["Date", "Close"])

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
        if len(closes) < 2:
            continue

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
            "Ticker": str(tkr),
            "Region": str(reg),
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
Return STRICT JSON only, no prose, no markdown.

Schema:
{{
  "forecasts": [
    {{"Ticker":"...", "Region":"RO|EU|US", "ER_Pct": 0.0}}
  ]
}}

Date: {as_of.isoformat()}
HorizonDays: {horizon_days}

Rules:
- Conservative magnitudes, typical weekly ER within +/-10.
- If weak signal, ER close to 0.
- Use only the provided price-based context.

HOLDINGS:
{json.dumps(hold_records, indent=2)}

TICKER_FEATURES:
{json.dumps(feat_records, indent=2)}

REGION_SUMMARY:
{json.dumps(reg_records, indent=2)}
""".strip()
    return prompt


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Parsing defensiv: încearcă json direct; dacă nu merge, caută primul obiect { ... }.
    """
    if not text:
        return None

    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        chunk = t[first:last + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Returnează:
      Ticker, Region, ER_Pct
      EngineStatus: OK / FALLBACK_ZERO
      LastError: motiv scurt la fallback
    """
    if holdings.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "ER_Pct", "EngineStatus", "LastError"])

    holdings = holdings[["Ticker", "Region"]].drop_duplicates().copy()
    holdings["Ticker"] = holdings["Ticker"].astype(str)
    holdings["Region"] = holdings["Region"].astype(str)

    ticker_features = _compute_ticker_features(prices_history, holdings)
    region_summary = _compute_region_summary(ticker_features)
    prompt = _build_prompt(holdings, ticker_features, region_summary, as_of, horizon_days)

    system_msg = (
        f"You are {settings.MODEL_VERSION_TAG}, a systematic multi-region equity forecaster. "
        f"Output must be strict JSON matching the provided schema."
    )

    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=900,
            )

            content = resp.choices[0].message.content if resp.choices else ""
            parsed = _extract_json_object(content)
            if not parsed or "forecasts" not in parsed:
                raise ValueError("Model output is not valid JSON or missing 'forecasts'.")

            df = pd.DataFrame(parsed.get("forecasts", []))
            if df.empty:
                raise ValueError("Empty forecasts from model output.")

            for col in ["Ticker", "Region", "ER_Pct"]:
                if col not in df.columns:
                    raise ValueError(f"Missing column in forecasts: {col}")

            df["Ticker"] = df["Ticker"].astype(str)
            df["Region"] = df["Region"].astype(str)
            df["ER_Pct"] = pd.to_numeric(df["ER_Pct"], errors="coerce").fillna(0.0)

            merged = holdings.merge(df[["Ticker", "Region", "ER_Pct"]], on=["Ticker", "Region"], how="left")
            merged["ER_Pct"] = merged["ER_Pct"].fillna(0.0)
            merged["EngineStatus"] = "OK"
            merged["LastError"] = ""
            return merged

        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            sleep_s = 2 ** attempt
            print(f"[STOCKD] OpenAI call failed (attempt {attempt}/3): {last_err}. Retrying in {sleep_s}s")
            time.sleep(sleep_s)

    print(f"[STOCKD] ERROR calling OpenAI after retries: {last_err}")
    fb = holdings.copy()
    fb["ER_Pct"] = 0.0
    fb["EngineStatus"] = "FALLBACK_ZERO"
    fb["LastError"] = last_err or "unknown"
    return fb
