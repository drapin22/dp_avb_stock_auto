# stockd/entity_profiles.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import yfinance as yf

from stockd import settings


STOPWORDS = {
    "inc", "corp", "corporation", "ltd", "plc", "sa", "s.a.", "ag", "nv", "the",
    "group", "holding", "holdings", "company", "co", "class", "ordinary", "shares",
    "common", "stock"
}


@dataclass
class EntityProfile:
    ticker: str
    region: str
    company_name: str = ""
    exchange: str = ""
    country: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    keywords: List[str] = None  # tokens useful for news filtering
    source: str = ""            # "yfinance" / "holdings" / "fallback"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["keywords"] = self.keywords or []
        return d


def _cache_path() -> Path:
    return settings.DATA_DIR / "entity_profiles.json"


def _load_cache() -> Dict[str, Any]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _key(ticker: str, region: str) -> str:
    return f"{ticker.upper()}|{region.upper()}"


def _normalize_tokens(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"[^\w\s\.-]", " ", text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    parts = []
    for p in t.split(" "):
        p = p.strip(".-")
        if not p or len(p) < 3:
            continue
        if p in STOPWORDS:
            continue
        parts.append(p)
    # unique, preserve order
    seen = set()
    out = []
    for x in parts:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:10]


def _profile_from_yfinance(ticker: str, region: str) -> EntityProfile:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    name = (
        info.get("longName")
        or info.get("shortName")
        or info.get("displayName")
        or ""
    )

    prof = EntityProfile(
        ticker=ticker.upper(),
        region=region.upper(),
        company_name=str(name or "").strip(),
        exchange=str(info.get("exchange") or "").strip(),
        country=str(info.get("country") or "").strip(),
        sector=str(info.get("sector") or "").strip(),
        industry=str(info.get("industry") or "").strip(),
        currency=str(info.get("currency") or "").strip(),
        keywords=[],
        source="yfinance",
    )

    # keywords: company name tokens + a couple of high signal tokens (sector, industry)
    kw = []
    kw += _normalize_tokens(prof.company_name)
    kw += _normalize_tokens(prof.sector)
    kw += _normalize_tokens(prof.industry)

    # Keep only a few strong ones
    prof.keywords = kw[:12]
    return prof


def _profile_fallback(ticker: str, region: str) -> EntityProfile:
    # conservative fallback: avoid inventing; use ticker only
    prof = EntityProfile(
        ticker=ticker.upper(),
        region=region.upper(),
        company_name="",
        exchange="",
        country="",
        sector="",
        industry="",
        currency="",
        keywords=[],
        source="fallback",
    )
    return prof


def get_entity_profile(ticker: str, region: str, refresh: bool = False) -> EntityProfile:
    """
    Returns a cached profile if available. Otherwise builds it using yfinance (US/EU),
    fallback for RO if yfinance doesn't help.
    """
    ticker_u = (ticker or "").upper().strip()
    region_u = (region or "").upper().strip()
    k = _key(ticker_u, region_u)

    cache = _load_cache()
    if not refresh and k in cache:
        try:
            d = cache[k]
            return EntityProfile(
                ticker=d.get("ticker", ticker_u),
                region=d.get("region", region_u),
                company_name=d.get("company_name", ""),
                exchange=d.get("exchange", ""),
                country=d.get("country", ""),
                sector=d.get("sector", ""),
                industry=d.get("industry", ""),
                currency=d.get("currency", ""),
                keywords=d.get("keywords", []) or [],
                source=d.get("source", "cache"),
            )
        except Exception:
            pass

    prof: EntityProfile
    if region_u in {"US", "EU"}:
        prof = _profile_from_yfinance(ticker_u, region_u)
        # if name is missing, fallback
        if not prof.company_name:
            prof = _profile_fallback(ticker_u, region_u)
    else:
        # RO: yfinance often doesn't know BVB symbols in your format
        prof = _profile_fallback(ticker_u, region_u)

    cache[k] = prof.to_dict()
    _save_cache(cache)
    return prof


def warm_entity_profiles(universe: pd.DataFrame, refresh: bool = False) -> None:
    """
    universe: DataFrame with columns Ticker, Region
    """
    if universe is None or universe.empty:
        return
    for _, r in universe.dropna(subset=["Ticker", "Region"]).iterrows():
        get_entity_profile(str(r["Ticker"]), str(r["Region"]), refresh=refresh)
