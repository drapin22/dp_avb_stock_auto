# stockd/mentor.py
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from stockd import settings
from stockd.news_rss import fetch_headlines_for_ticker

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_OK = False


RELEVANCE_MIN = 0.55  # must match news_rss.py


def _overrides_path() -> Path:
    return getattr(settings, "MENTOR_OVERRIDES_FILE", settings.DATA_DIR / "mentor_overrides.json")


def _raw_path(week_end: date) -> Path:
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return settings.REPORTS_DIR / f"mentor_raw_{week_end.isoformat()}.txt"


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    t = text.strip()
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    def _loads_once(s: str) -> Any:
        return json.loads(s)

    try:
        obj = _loads_once(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            s2 = obj.strip()
            try:
                obj2 = _loads_once(s2)
                if isinstance(obj2, dict):
                    return obj2
            except Exception:
                pass
            a2 = s2.find("{")
            b2 = s2.rfind("}")
            if a2 != -1 and b2 != -1 and b2 > a2:
                chunk2 = s2[a2:b2 + 1]
                try:
                    obj2 = _loads_once(chunk2)
                    if isinstance(obj2, dict):
                        return obj2
                except Exception:
                    return None
        return None
    except Exception:
        pass

    a = t.find("{")
    b = t.rfind("}")
    if a != -1 and b != -1 and b > a:
        chunk = t[a:b + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                try:
                    obj2 = json.loads(obj)
                    if isinstance(obj2, dict):
                        return obj2
                except Exception:
                    return None
        except Exception:
            return None

    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _write_overrides(payload: dict) -> None:
    p = _overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, date):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]

    if isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)


def _build_postmortem_md(worst: pd.DataFrame, overrides: dict, week_end: date, coverage: dict) -> Path:
    md_path = settings.REPORTS_DIR / f"mentor_postmortem_{week_end.isoformat()}.md"

    items = overrides.get("items", []) or []
    idx = {(it.get("Ticker"), it.get("Region")): it for it in items}

    lines: List[str] = []
    lines.append(f"# StockD Mentor Postmortem ({week_end.isoformat()})\n")
    lines.append("Mentor uses ONLY provided headlines. If coverage is low, outputs UNKNOWN.\n")
    lines.append("| Ticker | Region | Pred% | Real% | AbsErr pp | NewsCov | Cause | Conf | Overrides | Evidence |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---|---|")

    for _, r in worst.iterrows():
        t = str(r.get("Ticker", ""))
        reg = str(r.get("Region", ""))
        pred = float(r.get("Model_ER_Pct", 0.0))
        real = float(r.get("Realized_Pct", 0.0))
        ab = float(r.get("AbsError_Pct", 0.0))

        cov = coverage.get(f"{t}|{reg}", 0)

        it = idx.get((t, reg), {})
        cause = str(it.get("cause", "UNKNOWN"))
        conf = float(it.get("confidence_0_1", 0.0))

        ov = []
        if "clip_pct" in it:
            ov.append(f"clip={float(it['clip_pct']):.1f}")
        if "multiplier_cap" in it:
            ov.append(f"cap={float(it['multiplier_cap']):.2f}")
        ov_txt = ", ".join(ov)

        ev = it.get("evidence_headlines", []) or []
        ev_txt = "; ".join([str(x) for x in ev[:2]])[:160]

        lines.append(f"| {t} | {reg} | {pred:+.2f} | {real:+.2f} | {ab:.2f} | {cov} | {cause} | {conf:.2f} | {ov_txt} | {ev_txt} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _call_mentor_llm(client: OpenAI, model_name: str, system: str, user_payload: dict) -> str:
    user_text = json.dumps(_to_json_safe(user_payload), ensure_ascii=False)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content if resp.choices else ""
    except TypeError:
        pass
    except Exception:
        pass

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
        max_tokens=1400,
    )
    return resp.choices[0].message.content if resp.choices else ""


def run_mentor(eval_df: pd.DataFrame, week_end: date, top_n: int = 8, headlines_days: int = 12) -> Dict[str, Any]:
    import os

    if eval_df is None or eval_df.empty:
        overrides = {"status": "NO_EVAL", "items": [], "global_notes": "No eval rows provided."}
        _write_overrides(overrides)
        return {"status": "NO_EVAL"}

    df = eval_df.copy()
    for c in ["AbsError_Pct", "Error_Pct", "Model_ER_Pct", "Realized_Pct"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    df = df.dropna(subset=["Ticker", "Region", "AbsError_Pct", "Model_ER_Pct", "Realized_Pct"])
    if df.empty:
        overrides = {"status": "NO_VALID_ROWS", "items": [], "global_notes": "All rows invalid after cleaning."}
        _write_overrides(overrides)
        return {"status": "NO_VALID_ROWS"}

    worst = df.sort_values("AbsError_Pct", ascending=False).head(top_n).copy()

    # build headlines payload with relevance filter
    head_payload: List[Dict[str, Any]] = []
    coverage: Dict[str, int] = {}

    for _, r in worst.iterrows():
        t = str(r["Ticker"])
        reg = str(r["Region"])
        h = fetch_headlines_for_ticker(t, reg, since_days=headlines_days, max_items=10)

        if h is None or h.empty:
            coverage[f"{t}|{reg}"] = 0
            continue

        # if Relevance column exists, keep only above threshold
        if "Relevance" in h.columns:
            h = h[pd.to_numeric(h["Relevance"], errors="coerce").fillna(0.0) >= RELEVANCE_MIN]

        coverage[f"{t}|{reg}"] = int(len(h))

        for _, hr in h.head(6).iterrows():
            head_payload.append({
                "Ticker": t,
                "Region": reg,
                "PublishedAt": hr.get("PublishedAt", ""),
                "Headline": hr.get("Headline", ""),
                "Link": hr.get("Link", ""),
                "Query": hr.get("Query", ""),
                "Relevance": float(hr.get("Relevance", 1.0)) if "Relevance" in hr else 1.0,
            })

    api_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if not _OPENAI_OK or not api_key_present:
        overrides = {
            "status": "SKIPPED_NO_OPENAI",
            "items": [
                {"Ticker": str(r["Ticker"]), "Region": str(r["Region"]), "cause": "UNKNOWN", "confidence_0_1": 0.0, "evidence_headlines": []}
                for _, r in worst.iterrows()
            ],
            "global_notes": "OpenAI not available or OPENAI_API_KEY missing.",
            "news_coverage": coverage,
        }
        _write_overrides(overrides)
        md_path = _build_postmortem_md(worst, overrides, week_end, coverage)
        return {"status": "SKIPPED_NO_OPENAI", "md_path": str(md_path)}

    worst_payload = worst[[
        "Ticker", "Region", "WeekStart", "TargetDate",
        "Model_ER_Pct", "Realized_Pct", "Error_Pct", "AbsError_Pct", "DirectionHit"
    ]].to_dict(orient="records")

    system = (
        "You are StockD Mentor. Diagnose forecast errors using ONLY the provided headlines and numeric data. "
        "If a ticker has 0 relevant headlines (coverage=0), set cause=UNKNOWN and confidence_0_1=0.0. "
        "Return STRICT JSON object only (no markdown)."
    )

    user_payload = {
        "task": "Post-mortem diagnosis of why forecast was wrong, plus conservative guardrails for next time.",
        "allowed_causes": [
            "EARNINGS", "MACRO", "POLICY_REGULATION", "SECTOR", "COMPANY_NEWS",
            "LIQUIDITY_MICROSTRUCTURE", "FX_RATES", "TECHNICALS", "UNKNOWN"
        ],
        "rules": {
            "use_only_provided_headlines": True,
            "do_not_invent_events": True,
            "if_no_headlines_then_unknown": True,
            "overrides_must_be_conservative": True,
            "override_limits": {"clip_pct_range": [1.0, 10.0], "multiplier_cap_range": [0.25, 2.50]},
        },
        "news_coverage": coverage,
        "worst_cases": _to_json_safe(worst_payload),
        "headlines": _to_json_safe(head_payload),
        "output_schema": {
            "diagnostics": [
                {
                    "Ticker": "X",
                    "Region": "RO",
                    "cause": "UNKNOWN",
                    "confidence_0_1": 0.0,
                    "evidence_headlines": ["..."],
                    "safe_overrides": {"clip_pct": 5.0, "multiplier_cap": 1.2},
                    "what_to_learn_next_time": ["..."]
                }
            ],
            "global_notes": "..."
        }
    }

    model_name = getattr(settings, "COACH_MODEL_NAME", None) or settings.MODEL_NAME
    client = OpenAI()

    content1 = _call_mentor_llm(client, model_name, system, user_payload)
    _raw_path(week_end).write_text((content1 or "") + "\n", encoding="utf-8")
    parsed = _extract_json_object(content1 or "")

    if not parsed or "diagnostics" not in parsed:
        hard_system = (
            "Return ONLY a valid JSON object. No markdown. No extra keys. "
            "If unsure, set cause=UNKNOWN and confidence_0_1=0.0."
        )
        content2 = _call_mentor_llm(client, model_name, hard_system, user_payload)
        _raw_path(week_end).write_text(((content1 or "") + "\n\n--- RETRY ---\n\n" + (content2 or "")), encoding="utf-8")
        parsed = _extract_json_object(content2 or "")

    if not parsed or "diagnostics" not in parsed:
        overrides = {"status": "LLM_INVALID", "items": [], "global_notes": ((content1 or "")[:800]), "news_coverage": coverage}
        _write_overrides(overrides)
        md_path = _build_postmortem_md(worst, overrides, week_end, coverage)
        return {"status": "LLM_INVALID", "md_path": str(md_path)}

    overrides: Dict[str, Any] = {
        "status": "OK",
        "items": [],
        "global_notes": str(parsed.get("global_notes", ""))[:800],
        "news_coverage": coverage,
    }

    for d in parsed.get("diagnostics", []):
        t = str(d.get("Ticker", "")).strip()
        reg = str(d.get("Region", "")).strip()

        # enforce unknown if no headlines
        cov = int(coverage.get(f"{t}|{reg}", 0))
        if cov <= 0:
            overrides["items"].append({
                "Ticker": t,
                "Region": reg,
                "cause": "UNKNOWN",
                "confidence_0_1": 0.0,
                "evidence_headlines": [],
                "what_to_learn_next_time": ["Increase news coverage signal quality or accept UNKNOWN."]
            })
            continue

        cause = str(d.get("cause", "UNKNOWN")).strip()
        conf = _clamp(float(d.get("confidence_0_1", 0.2)), 0.0, 1.0)

        so = d.get("safe_overrides", {}) or {}
        item = {"Ticker": t, "Region": reg, "cause": cause, "confidence_0_1": conf}
        item["evidence_headlines"] = (d.get("evidence_headlines", []) or [])[:4]

        if "clip_pct" in so:
            item["clip_pct"] = _clamp(float(so["clip_pct"]), 1.0, 10.0)
        if "multiplier_cap" in so:
            item["multiplier_cap"] = _clamp(float(so["multiplier_cap"]), 0.25, 2.50)

        overrides["items"].append(item)

    _write_overrides(overrides)
    md_path = _build_postmortem_md(worst, overrides, week_end, coverage)
    return {"status": "OK", "md_path": str(md_path)}


def run_mentor_diagnostics(eval_df: pd.DataFrame, week_end: date, **kwargs) -> Dict[str, Any]:
    return run_mentor(eval_df, week_end=week_end, **kwargs)
