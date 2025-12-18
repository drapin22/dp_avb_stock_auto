# stockd/mentor.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from stockd import settings


@dataclass
class MentorResult:
    status: str
    error: str
    md_path: Optional[str] = None
    overrides_path: Optional[str] = None


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _safe_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    # extract first JSON object in text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _call_openai_json(prompt: str) -> tuple[Optional[dict], str]:
    """
    Returns (json_obj, error_str).
    Works across OpenAI library differences by using the simplest call pattern possible.
    """
    api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
    if not api_key:
        return None, "OPENAI_API_KEY missing"

    model = getattr(settings, "STOCKD_COACH_MODEL_NAME", None) or getattr(settings, "STOCKD_MODEL_NAME", "gpt-4.1-mini")

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)

        # Use Responses API without response_format (you hit issues with that earlier).
        resp = client.responses.create(
            model=model,
            input=prompt,
        )

        # Best-effort to get text
        text = ""
        try:
            text = getattr(resp, "output_text", "") or ""
        except Exception:
            text = ""
        if not text:
            # fallback parse
            try:
                text = str(resp)
            except Exception:
                text = ""

        obj = _safe_json_from_text(text)
        if obj is None:
            return None, "LLM returned non-JSON"
        return obj, ""

    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def _fetch_headlines(ticker: str, region: str, max_items: int = 5) -> List[str]:
    """
    Optional integration. If you have stockd/news_utils.py with a function, we use it.
    Otherwise we return empty list. This avoids AttributeError crashes.
    """
    try:
        from stockd.news_utils import fetch_headlines_for_ticker  # type: ignore
        out = fetch_headlines_for_ticker(ticker=ticker, region=region, limit=max_items)
        if isinstance(out, list):
            return [str(x) for x in out][:max_items]
        return []
    except Exception:
        return []


def _build_prompt(eval_df: pd.DataFrame, week_end: date) -> str:
    """
    Mentor prompt: ask for structured JSON diagnostics + safe overrides.
    """
    df = eval_df.copy()

    # Select worst cases for postmortem
    for c in ["AbsError_Pct", "Error_Pct", "Model_ER_Pct", "Realized_Pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "AbsError_Pct" not in df.columns and "Error_Pct" in df.columns:
        df["AbsError_Pct"] = df["Error_Pct"].abs()

    # worst 8 rows
    if "AbsError_Pct" in df.columns:
        df = df.sort_values("AbsError_Pct", ascending=False).head(8)

    cases = []
    for _, r in df.iterrows():
        ticker = str(r.get("Ticker", "")).strip()
        region = str(r.get("Region", "")).strip()
        if not ticker or not region:
            continue
        headlines = _fetch_headlines(ticker, region, max_items=5)
        cases.append({
            "Ticker": ticker,
            "Region": region,
            "Model_ER_Pct": float(r.get("Model_ER_Pct", 0.0)) if pd.notna(r.get("Model_ER_Pct", 0.0)) else 0.0,
            "Realized_Pct": float(r.get("Realized_Pct", 0.0)) if pd.notna(r.get("Realized_Pct", 0.0)) else 0.0,
            "Error_Pct": float(r.get("Error_Pct", 0.0)) if pd.notna(r.get("Error_Pct", 0.0)) else 0.0,
            "Headlines": headlines,
        })

    schema_desc = """
Return STRICT JSON with keys:
{
  "diagnostics": [
    {
      "Ticker": "string",
      "Region": "RO|EU|US",
      "cause": "EARNINGS|MACRO|POLICY_REGULATION|SECTOR|IDIOSYNCRATIC|LIQUIDITY|NEWS_SENTIMENT|OTHER",
      "confidence_0_1": 0.0-1.0,
      "evidence_headlines": ["string", ...],
      "what_to_learn_next_time": ["string", ...],
      "safe_overrides": {
        "clip_pct": number,           // max abs forecast to allow next time for this ticker
        "multiplier_cap": number      // cap on scaling factors
      }
    }
  ],
  "global_notes": "short string"
}
"""

    return (
        f"You are StockD Mentor. Week end: {week_end.isoformat()}.\n"
        "We evaluate forecast errors and want to learn and improve.\n\n"
        "CASES (worst errors):\n"
        f"{json.dumps(cases, ensure_ascii=False)}\n\n"
        "TASK:\n"
        "1) Diagnose likely causes for each case using provided headlines (if any) and market common sense.\n"
        "2) Propose SAFE overrides (clip_pct, multiplier_cap) that reduce future blowups.\n"
        "3) Keep it conservative. If evidence is weak, say OTHER with low confidence.\n\n"
        f"{schema_desc}"
    )


def run_mentor(eval_df: pd.DataFrame, week_end: date) -> Dict[str, Any]:
    """
    Produces:
    - data/mentor_overrides.json
    - reports/mentor_postmortem_<week_end>.md
    Always returns a dict, never raises (avoid pipeline break).
    """
    overrides_path = Path(getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json"))
    md_path = settings.REPORTS_DIR / f"mentor_postmortem_{week_end.isoformat()}.md"

    # Default fallback payload
    fallback = {
        "status": "LLM_INVALID",
        "items": [],
        "global_notes": "",
    }

    try:
        if eval_df is None or eval_df.empty:
            _write_json(overrides_path, fallback)
            _write_text(md_path, "# Mentor postmortem\n\nNo eval rows.\n")
            return {"status": "LLM_INVALID", "error": "No eval rows", "md_path": str(md_path), "overrides_path": str(overrides_path)}

        prompt = _build_prompt(eval_df, week_end)
        obj, err = _call_openai_json(prompt)

        if obj is None:
            _write_json(overrides_path, fallback)
            _write_text(md_path, f"# Mentor postmortem\n\nLLM invalid.\n\nError: {err}\n")
            return {"status": "LLM_INVALID", "error": err, "md_path": str(md_path), "overrides_path": str(overrides_path)}

        # Normalize expected keys
        diagnostics = obj.get("diagnostics", [])
        if not isinstance(diagnostics, list):
            diagnostics = []

        global_notes = obj.get("global_notes", "")
        if not isinstance(global_notes, str):
            global_notes = ""

        payload = {
            "status": "OK",
            "items": diagnostics,
            "global_notes": global_notes,
        }

        _write_json(overrides_path, payload)

        # Markdown report
        lines = []
        lines.append("# Mentor postmortem")
        lines.append("")
        lines.append(f"Week end: {week_end.isoformat()}")
        lines.append("")
        if global_notes:
            lines.append("## Global notes")
            lines.append(global_notes)
            lines.append("")
        lines.append("## Diagnostics")
        if not diagnostics:
            lines.append("No diagnostics.")
        else:
            for d in diagnostics:
                t = d.get("Ticker", "")
                r = d.get("Region", "")
                cause = d.get("cause", "")
                conf = d.get("confidence_0_1", "")
                lines.append(f"- **{t} ({r})** cause={cause}, conf={conf}")
                ev = d.get("evidence_headlines", [])
                if isinstance(ev, list) and ev:
                    for h in ev[:5]:
                        lines.append(f"  - {h}")
                learn = d.get("what_to_learn_next_time", [])
                if isinstance(learn, list) and learn:
                    lines.append("  - Learn next time:")
                    for x in learn[:5]:
                        lines.append(f"    - {x}")
                so = d.get("safe_overrides", {})
                if isinstance(so, dict) and so:
                    lines.append(f"  - Overrides: clip_pct={so.get('clip_pct')}, multiplier_cap={so.get('multiplier_cap')}")
        lines.append("")

        _write_text(md_path, "\n".join(lines))
        return {"status": "OK", "error": "", "md_path": str(md_path), "overrides_path": str(overrides_path)}

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        _write_json(overrides_path, fallback)
        _write_text(md_path, f"# Mentor postmortem\n\nMentor crashed.\n\nError: {err}\n")
        return {"status": "LLM_INVALID", "error": err, "md_path": str(md_path), "overrides_path": str(overrides_path)}
