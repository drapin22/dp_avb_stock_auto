# stockd/mentor.py
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from openai import OpenAI

from stockd import settings
from stockd.ticker_aliases import load_aliases, save_aliases, auto_enrich_aliases, build_query


def _json_safe(obj):
    """
    Ensures any pandas Timestamp is serializable.
    """
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (date,)):
        return obj.isoformat()
    return str(obj)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_mentor(
    eval_df: pd.DataFrame,
    week_end: str,
    max_items: int = 12,
) -> Dict[str, Any]:
    """
    Produces:
      data/mentor_overrides.json
      reports/mentor_postmortem_<week_end>.md
    Does not change forecasts directly; it writes safe rules and diagnostics.
    """
    out_overrides = settings.DATA_DIR / "mentor_overrides.json"
    out_md = settings.REPORTS_DIR / f"mentor_postmortem_{week_end}.md"

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Basic validation
    needed = {"Ticker", "Region", "ER_Pct", "RealReturnPct", "ErrorPct"}
    if not needed.issubset(set(eval_df.columns)):
        payload = {"status": "INPUT_INVALID", "items": [], "global_notes": ""}
        out_overrides.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text("Mentor: INPUT_INVALID\n", encoding="utf-8")
        return payload

    df = eval_df.copy()
    df["abs_error"] = df["ErrorPct"].abs()
    worst = df.sort_values("abs_error", ascending=False).head(max_items).reset_index(drop=True)

    # build / enrich aliases for ambiguous tickers automatically
    aliases = load_aliases()
    for _, r in worst.iterrows():
        aliases = auto_enrich_aliases(str(r["Ticker"]), str(r["Region"]), aliases)
    save_aliases(aliases)

    items: List[Dict[str, Any]] = []
    for _, r in worst.iterrows():
        ticker = str(r["Ticker"])
        region = str(r["Region"])
        q = build_query(ticker, region, aliases)
        items.append(
            {
                "Ticker": ticker,
                "Region": region,
                "ER_Pct": float(r["ER_Pct"]),
                "RealReturnPct": float(r["RealReturnPct"]),
                "ErrorPct": float(r["ErrorPct"]),
                "NewsQuery": q,
            }
        )

    system = (
        "You are a strict trading model mentor. "
        "Your job is to diagnose why forecasts were wrong and propose ONLY safe, actionable adjustments. "
        "Do not hallucinate headlines. If you cannot justify, mark as unknown."
    )

    user = {
        "week_end": week_end,
        "worst_errors": items,
        "required_output": {
            "safe_overrides": {"clip_pct": "number", "multiplier_cap": "number"},
            "diagnostics": [
                {
                    "Ticker": "string",
                    "Region": "string",
                    "cause": "one of: EARNINGS, MACRO, POLICY_REGULATION, COMPANY_NEWS, SENTIMENT, LIQUIDITY, UNKNOWN",
                    "confidence_0_1": "number",
                    "what_to_learn_next_time": ["string", "string"],
                    "query_fix": "optional improved news query for this ticker",
                }
            ],
        },
    }

    status = "OK"
    global_notes_obj: Dict[str, Any] = {}
    raw_text = ""

    try:
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        c = _client()
        resp = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, default=_json_safe, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw_text = resp.choices[0].message.content or ""
        global_notes_obj = json.loads(raw_text)
    except Exception as e:
        status = "LLM_INVALID"
        global_notes_obj = {
            "safe_overrides": {"clip_pct": 6.0, "multiplier_cap": 1.3},
            "diagnostics": [],
            "error": str(e),
        }

    payload = {
        "status": status,
        "items": items,
        "global_notes": json.dumps(global_notes_obj, ensure_ascii=False),
    }

    out_overrides.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown postmortem for humans
    md_lines = []
    md_lines.append(f"# StockD mentor postmortem\n")
    md_lines.append(f"Week end: {week_end}\n")
    md_lines.append(f"Status: {status}\n\n")
    md_lines.append("## Worst errors input\n")
    for it in items:
        md_lines.append(
            f"- {it['Ticker']} {it['Region']} raw={it['ER_Pct']:+.2f}% real={it['RealReturnPct']:+.2f}% err={it['ErrorPct']:+.2f}%"
        )
    md_lines.append("\n## Mentor output\n")
    md_lines.append("```json")
    md_lines.append(json.dumps(global_notes_obj, ensure_ascii=False, indent=2))
    md_lines.append("```")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    return payload
