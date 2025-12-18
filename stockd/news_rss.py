# stockd/news_rss.py
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import requests

from stockd.entity_profiles import get_entity_profile


_FINANCE_HINTS = [
    "shares", "stock", "stocks", "earnings", "guidance", "revenue", "profit",
    "sec", "ipo", "dividend", "buyback", "acquisition", "merger", "results"
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _word_boundary_contains(text: str, token: str) -> bool:
    if not token:
        return False
    # token exact as word, case-insensitive
    return re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE) is not None


def _parse_rss_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _rss_search(query: str, max_items: int = 30) -> List[Dict[str, str]]:
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    xml = r.text

    items: List[Dict[str, str]] = []
    for block in xml.split("<item>")[1:]:
        if len(items) >= max_items:
            break

        def _tag(tag: str) -> str:
            a = block.find(f"<{tag}>")
            b = block.find(f"</{tag}>")
            if a == -1 or b == -1:
                return ""
            return block[a + len(tag) + 2 : b].strip()

        title = _tag("title")
        link = _tag("link")
        pub = _tag("pubDate")
        items.append({"Headline": title, "Link": link, "PublishedAt": pub, "Query": query})

    return items


def _build_queries(ticker: str, region: str) -> List[str]:
    prof = get_entity_profile(ticker, region)

    # Prefer company name queries when available (fixes PATH-like ambiguity)
    queries: List[str] = []

    if prof.company_name:
        # quoted company name + finance hints
        queries.append(f"\"{prof.company_name}\" stock")
        queries.append(f"\"{prof.company_name}\" earnings OR results")
        # also include ticker in a constrained way
        queries.append(f"\"{prof.company_name}\" ({ticker})")
    else:
        # fallback
        if region.upper() == "RO":
            queries.append(f"{ticker} BVB")
        else:
            queries.append(f"{ticker} stock")

    return queries[:4]


def _relevance_score(headline: str, ticker: str, region: str) -> float:
    """
    Deterministic relevance filter.
    Returns score 0..1. Keep if >= 0.55.
    """
    h = headline or ""
    hn = _normalize(h)
    prof = get_entity_profile(ticker, region)

    score = 0.0

    # Strong signals: company name tokens in headline
    if prof.company_name:
        # require at least one meaningful token from company keywords
        hit_kw = 0
        for kw in (prof.keywords or [])[:8]:
            if kw and kw in hn:
                hit_kw += 1
        if hit_kw >= 1:
            score += 0.55
        if hit_kw >= 2:
            score += 0.15

        # exact company name (rare but strong)
        if _normalize(prof.company_name) in hn:
            score += 0.25

    # Ticker mention as word boundary (helps but not sufficient alone for ambiguous tickers)
    if _word_boundary_contains(h, ticker.upper()):
        score += 0.20

    # Finance context words
    for w in _FINANCE_HINTS:
        if w in hn:
            score += 0.05
            break

    # Penalize extremely generic headlines if they lack company signals
    if score < 0.55:
        generic_penalty_words = ["path forward", "path ahead", "a path", "the path", "forward path"]
        for gp in generic_penalty_words:
            if gp in hn:
                score -= 0.30
                break

    return max(0.0, min(1.0, score))


def fetch_headlines_for_ticker(
    ticker: str,
    region: str,
    since_days: int = 14,
    max_items: int = 12,
) -> pd.DataFrame:
    ticker_u = (ticker or "").upper().strip()
    region_u = (region or "").upper().strip()

    queries = _build_queries(ticker_u, region_u)

    all_items: List[Dict[str, str]] = []
    for q in queries:
        try:
            all_items.extend(_rss_search(q, max_items=max_items * 4))
        except Exception:
            continue

    if not all_items:
        return pd.DataFrame(columns=["Ticker", "Region", "PublishedAt", "Headline", "Link", "Query", "Relevance"])

    df = pd.DataFrame(all_items).drop_duplicates(subset=["Headline", "Link"])

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=since_days)

    df["PublishedAtDT"] = df["PublishedAt"].apply(_parse_rss_date)
    df = df[df["PublishedAtDT"].notna()]
    df = df[df["PublishedAtDT"] >= cutoff]

    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Region", "PublishedAt", "Headline", "Link", "Query", "Relevance"])

    df["Relevance"] = df["Headline"].apply(lambda x: _relevance_score(str(x), ticker_u, region_u))

    # Keep only highly relevant items. If none, return empty (better than garbage).
    df = df[df["Relevance"] >= 0.55]

    df["Ticker"] = ticker_u
    df["Region"] = region_u
    df = df.sort_values(["Relevance", "PublishedAtDT"], ascending=[False, False]).head(max_items)

    return df[["Ticker", "Region", "PublishedAt", "Headline", "Link", "Query", "Relevance"]].reset_index(drop=True)
