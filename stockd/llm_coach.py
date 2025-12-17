# stockd/llm_coach.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
from openai import OpenAI

from stockd import settings

client = OpenAI()


def _extract_json_object(text: str) -> Optional[dict]:
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


def coach_calibration_suggestions(
    eval_summary: pd.DataFrame,
    worst_rows: pd.DataFrame,
    current_calib: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returnează sugestii JSON (nu aplică nimic singur).
    Sugestiile sunt ulterior validate/clamp-uite în calibration.py.
    """

    # micșorăm payload-ul ca să fie stabil
    summary_payload = eval_summary.to_dict(orient="records") if not eval_summary.empty else []
    worst_payload = worst_rows.to_dict(orient="records") if not worst_rows.empty else []

    system = (
        "You are a calibration coach for a quantitative forecasting system. "
        "You do NOT invent market predictions. "
        "You only recommend conservative calibration parameters from error statistics. "
        "Output STRICT JSON only."
    )

    # schema de output pe care o cerem (nu depindem de response_format)
    user = {
        "task": "Recommend conservative calibration updates based on evaluation stats.",
        "constraints": {
            "regions": ["RO", "EU", "US"],
            "clip_pct_range": [1.0, 10.0],
            "alpha_range": [0.05, 0.40],
            "max_multiplier_change_per_week": 0.15,
            "max_bias_change_per_week": 0.50
        },
        "current_calibration": current_calib,
        "eval_summary": summary_payload,
        "worst_examples": worst_payload,
        "output_schema": {
            "type": "object",
            "properties": {
                "alpha": {"type": "number"},
                "regions": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "clip_pct": {"type": "number"},
                            "notes": {"type": "string"}
                        },
                        "required": ["clip_pct"]
                    }
                },
                "ticker_overrides": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Ticker": {"type": "string"},
                            "Region": {"type": "string"},
                            "multiplier_cap": {"type": "number"},
                            "notes": {"type": "string"}
                        },
                        "required": ["Ticker", "Region", "multiplier_cap"]
                    }
                },
                "notes": {"type": "string"}
            },
            "required": ["alpha", "regions"]
        }
    }

    model_name = getattr(settings, "COACH_MODEL_NAME", None) or settings.MODEL_NAME

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    content = resp.choices[0].message.content if resp.choices else ""
    parsed = _extract_json_object(content)
    if not parsed:
        return {"_coach_status": "EMPTY_OR_INVALID", "raw": content[:500]}

    parsed["_coach_status"] = "OK"
    return parsed
