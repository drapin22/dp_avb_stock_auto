from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from stockd import settings
from stockd.news import fetch_headlines_for_ticker


def _safe_default(obj: Any) -> str:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _call_llm_json(prompt: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        try:
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            txt = resp.choices[0].message.content or ""
            return json.loads(txt)
        except Exception:
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown, no commentary."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            txt = resp.choices[0].message.content or ""
            return json.loads(txt)

    except Exception as e:
        return {"status": "LLM_ERROR", "error": str(e)}


def run_mentor(eval_df: pd.DataFrame, week_end: pd.Timestamp) -> Dict[str, Any]:
    try:
        if eval_df is None or eval_df.empty:
            return {"status": "EMPTY", "items": [], "safe_overrides": {"clip_pct": 8.0, "multiplier_cap": 1.5}}

        worst = eval_df.sort_values("abs_error_pp", ascending=False).head(8).copy()

        diagnostics = []
        for _, r in worst.iterrows():
            ticker = str(r["Ticker"])
            region = str(r["Region"])
            pred = float(r["PredictedReturnPct"])
            real = float(r["RealizedReturnPct"])
            headlines = fetch_headlines_for_ticker(ticker, region=region, days=10, max_items=6)

            diagnostics.append(
                {
                    "Ticker": ticker,
                    "Region": region,
                    "PredictedReturnPct": pred,
                    "RealizedReturnPct": real,
                    "AbsErrorPP": float(abs(pred - real)),
                    "Headlines": headlines,
                }
            )

        prompt = json.dumps(
            {
                "task": "You are a trading model mentor. Diagnose why predictions were wrong and propose SAFE adjustments only.",
                "constraints": [
                    "Use ONLY the provided headlines as evidence.",
                    "If headlines are ambiguous or not clearly about the company, say AMBIGUOUS_NEWS and lower confidence.",
                    "Return JSON with keys: status, safe_overrides, items, global_notes.",
                    "safe_overrides must include clip_pct (float) and multiplier_cap (float), conservative.",
                    "items is a list of objects with: Ticker, Region, cause, confidence_0_1, evidence_headlines (subset), what_to_learn_next_time (list).",
                ],
                "week_end": str(pd.Timestamp(week_end).date()),
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            default=_safe_default,
        )

        out = _call_llm_json(prompt)
        if not isinstance(out, dict) or "items" not in out:
            return {
                "status": "LLM_INVALID",
                "items": [],
                "safe_overrides": {"clip_pct": 8.0, "multiplier_cap": 1.5},
                "error": str(out)[:500],
            }

        out.setdefault("status", "OK")
        out.setdefault("safe_overrides", {"clip_pct": 8.0, "multiplier_cap": 1.5})
        return out

    except Exception as e:
        return {
            "status": "ERROR",
            "items": [],
            "safe_overrides": {"clip_pct": 8.0, "multiplier_cap": 1.5},
            "error": str(e),
        }
