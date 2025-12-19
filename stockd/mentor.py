from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from stockd import settings
from stockd.news import fetch_headlines_for_ticker


def _call_openai_json(prompt: str) -> Dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # chat.completions cu JSON
        resp = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No extra keys."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        txt = resp.choices[0].message.content or "{}"
        return json.loads(txt)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def propose_news_deltas(tickers: List[Dict], macro_snapshot: Dict) -> Dict:
    """
    tickers: [{Ticker, Region, base_er_pct}]
    Output: { ok, deltas: [{Ticker, Region, delta_pp, confidence_0_1, reason, ambiguous}] }
    """
    if not settings.OPENAI_API_KEY or not settings.ENABLE_LLM_NEWS_ADJ:
        return {"ok": False, "reason": "LLM_DISABLED"}

    pack = []
    for it in tickers:
        t = it["Ticker"]
        r = it["Region"]
        headlines = fetch_headlines_for_ticker(t, r, max_items=8)
        pack.append({
            "Ticker": t, "Region": r,
            "base_er_pct": float(it.get("base_er_pct", 0.0)),
            "headlines": headlines
        })

    prompt = json.dumps(
        {
            "task": "Adjust weekly return forecasts using ONLY provided headlines + macro snapshot. Conservative.",
            "constraints": [
                "Return delta_pp in [-MAX_DELTA, +MAX_DELTA] where MAX_DELTA is provided.",
                "If headlines are ambiguous or likely not about the company, set ambiguous=true and delta_pp=0.",
                "Do not invent events. Use only evidence headlines.",
                "If no headlines, delta_pp=0.",
                "Output JSON with: ok, deltas(list).",
            ],
            "MAX_DELTA": settings.MAX_NEWS_DELTA_PP,
            "macro_snapshot": macro_snapshot,
            "items": pack,
            "output_schema": {
                "ok": "bool",
                "deltas": [
                    {
                        "Ticker": "string",
                        "Region": "string",
                        "delta_pp": "float",
                        "confidence_0_1": "float",
                        "ambiguous": "bool",
                        "reason": "short string"
                    }
                ]
            }
        },
        ensure_ascii=False
    )

    out = _call_openai_json(prompt)
    if not isinstance(out, dict) or "deltas" not in out:
        return {"ok": False, "reason": "bad_llm_output", "raw": str(out)[:400]}
    return out


def postmortem_and_rules(eval_rows: List[Dict], macro_snapshot: Dict) -> Dict:
    """
    Output: safe rules only (caps, shrink, alias suggestions)
    """
    if not settings.OPENAI_API_KEY or not settings.ENABLE_MENTOR:
        return {"ok": False, "reason": "MENTOR_DISABLED"}

    prompt = json.dumps(
        {
            "task": "You are a strict mentor. Diagnose forecast errors and propose SAFE rule updates only.",
            "constraints": [
                "Do not propose per-ticker target prices.",
                "Only propose conservative rule parameters.",
                "Use only eval rows + macro snapshot.",
                "Return JSON with keys: ok, rules, insights.",
            ],
            "rules_schema": {
                "clip_abs_er_pct": "float (<=10)",
                "risk_off_shrink": "float in [0.4,1.0]",
                "ambiguous_news_policy": "string",
                "suggested_alias_updates": "list of {Ticker, Region, aliases[]}"
            },
            "macro_snapshot": macro_snapshot,
            "eval_rows": eval_rows[:50],  # limit
        },
        ensure_ascii=False
    )

    out = _call_openai_json(prompt)
    if not isinstance(out, dict) or "rules" not in out:
        return {"ok": False, "reason": "bad_llm_output", "raw": str(out)[:400]}
    return out
