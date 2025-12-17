# stockd/mentor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from openai import OpenAI

from stockd import settings
from stockd.news_rss import fetch_headlines_for_ticker

client = OpenAI()


def _mentor_overrides_path() -> Path:
    return getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")


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


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def run_mentor_diagnostics(
    eval_df: pd.DataFrame,
    top_n: int = 8,
    headlines_days: int = 12,
) -> Dict[str, Any]:
    """
    Ia cele mai mari greșeli din eval_df și încearcă să explice cauzele folosind headlines RSS.
    Scrie mentor_overrides.json cu reguli safe (clip/cap).
    Returnează dict cu rezumat + path override.
    """
    if eval_df is None or eval_df.empty:
        out = {"status": "NO_EVAL", "overrides_path": str(_mentor_overrides_path())}
        _mentor_overrides_path().write_text(json.dumps({"status": "NO_EVAL"}, indent=2), encoding="utf-8")
        return out

    df = eval_df.copy()
    df["AbsError_Pct"] = pd.to_numeric(df["AbsError_Pct"], errors="coerce")
    df["Error_Pct"] = pd.to_numeric(df["Error_Pct"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Region", "AbsError_Pct", "Error_Pct"])

    worst = df.sort_values("AbsError_Pct", ascending=False).head(top_n).copy()

    # colectăm headlines
    headlines_all = []
    for _, r in worst.iterrows():
        t = str(r["Ticker"])
        reg = str(r["Region"])
        h = fetch_headlines_for_ticker(t, reg, since_days=headlines_days, max_items=10)
        if not h.empty:
            headlines_all.append(h)

    headlines_df = pd.concat(headlines_all, ignore_index=True) if headlines_all else pd.DataFrame(
        columns=["PublishedAt", "Ticker", "Region", "Headline", "Link", "Source", "Query"]
    )

    # payload scurt și auditabil
    worst_payload = worst[[
        "Ticker", "Region", "WeekStart", "TargetDate",
        "Model_ER_Pct", "Realized_Pct", "Error_Pct", "AbsError_Pct", "DirectionHit"
    ]].to_dict(orient="records")

    head_payload = []
    if not headlines_df.empty:
        headlines_df["PublishedAt"] = headlines_df["PublishedAt"].astype(str)
        # limităm la 6 per ticker ca să nu explodeze promptul
        for (tkr, reg), g in headlines_df.groupby(["Ticker", "Region"], sort=False):
            for _, hr in g.head(6).iterrows():
                head_payload.append({
                    "Ticker": tkr,
                    "Region": reg,
                    "PublishedAt": hr["PublishedAt"],
                    "Headline": hr["Headline"],
                    "Source": hr.get("Source", ""),
                    "Link": hr.get("Link", ""),
                })

    system = (
        "You are StockD Mentor. "
        "Your job is post-mortem diagnosis of forecast errors using ONLY the provided headlines. "
        "If evidence is insufficient, answer UNKNOWN. "
        "Output STRICT JSON only. No prose."
    )

    user = {
        "task": "Diagnose why the forecast was wrong and propose safe guardrail overrides.",
        "allowed_causes": [
            "EARNINGS", "MACRO", "POLICY_REGULATION", "SECTOR", "COMPANY_NEWS",
            "LIQUIDITY_MICROSTRUCTURE", "FX_RATES", "TECHNICALS", "UNKNOWN"
        ],
        "rules": {
            "use_only_provided_headlines": True,
            "do_not_invent_events": True,
            "overrides_must_be_conservative": True,
            "override_limits": {
                "clip_pct_range": [1.0, 10.0],
                "multiplier_cap_range": [0.25, 2.50]
            }
        },
        "worst_cases": worst_payload,
        "headlines": head_payload,
        "output_schema": {
            "diagnostics": [
                {
                    "Ticker": "FP",
                    "Region": "RO",
                    "cause": "EARNINGS",
                    "confidence_0_1": 0.6,
                    "evidence_headlines": ["..."],
                    "what_to_learn_next_time": ["..."],
                    "safe_overrides": {
                        "clip_pct": 5.0,
                        "multiplier_cap": 1.2
                    }
                }
            ],
            "global_notes": "..."
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
        max_tokens=1200,
    )

    content = resp.choices[0].message.content if resp.choices else ""
    parsed = _extract_json_object(content)
    if not parsed or "diagnostics" not in parsed:
        overrides = {"status": "LLM_INVALID", "raw": content[:600]}
        _mentor_overrides_path().write_text(json.dumps(overrides, indent=2), encoding="utf-8")
        return {"status": "LLM_INVALID", "overrides_path": str(_mentor_overrides_path()), "raw": content[:600]}

    # construim overrides safe
    overrides: Dict[str, Any] = {"status": "OK", "items": []}
    for d in parsed.get("diagnostics", []):
        t = str(d.get("Ticker", "")).strip()
        r = str(d.get("Region", "")).strip()
        so = d.get("safe_overrides", {}) or {}
        clip_pct = so.get("clip_pct", None)
        mult_cap = so.get("multiplier_cap", None)

        item = {"Ticker": t, "Region": r}

        if clip_pct is not None:
            item["clip_pct"] = _clamp(float(clip_pct), 1.0, 10.0)
        if mult_cap is not None:
            item["multiplier_cap"] = _clamp(float(mult_cap), 0.25, 2.50)

        item["cause"] = str(d.get("cause", "UNKNOWN"))
        item["confidence_0_1"] = _clamp(float(d.get("confidence_0_1", 0.2)), 0.0, 1.0)
        item["evidence_headlines"] = (d.get("evidence_headlines", []) or [])[:4]
        overrides["items"].append(item)

    overrides["global_notes"] = str(parsed.get("global_notes", ""))[:600]

    _mentor_overrides_path().parent.mkdir(parents=True, exist_ok=True)
    _mentor_overrides_path().write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "OK",
        "overrides_path": str(_mentor_overrides_path()),
        "n_cases": int(len(worst)),
        "n_headlines": int(len(head_payload)),
        "global_notes": overrides.get("global_notes", ""),
    }
