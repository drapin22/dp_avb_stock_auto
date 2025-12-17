# stockd/news_rss.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import urllib.parse

import pandas as pd
import feedparser
import requests


def _google_news_rss_url(query: str, region: str) -> str:
    q = urllib.parse.quote(query)
    # setări simple pe regiuni
    if region == "RO":
        return f"https://news.google.com/rss/search?q={q}&hl=ro&gl=RO&ceid=RO:ro"
    if region == "EU":
        return f"https://news.google.com/rss/search?q={q}&hl=de&gl=DE&ceid=DE:de"
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_headlines_for_ticker(
    ticker: str,
    region: str,
    since_days: int = 10,
    max_items: int = 12,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    RSS headlines pentru query ticker (și variantă cu market context).
    Returnează:
      PublishedAt, Ticker, Region, Headline, Link, Source
    """
    ticker = str(ticker).strip()
    region = str(region).strip()

    # query-uri mici, robuste
    queries = [ticker]
    if region == "RO":
        queries.append(f"{ticker} BVB")
        queries.append(f"{ticker} rezultate financiare")
    elif region == "EU":
        queries.append(f"{ticker} earnings")
    else:
        queries.append(f"{ticker} earnings")

    rows = []
    cutoff = datetime.utcnow() - timedelta(days=since_days)

    for q in queries:
        url = _google_news_rss_url(q, region)
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
        except Exception:
            continue

        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            source = getattr(e, "source", {}).get("title") if isinstance(getattr(e, "source", None), dict) else ""

            # published parsing tolerant
            pub = None
            for key in ["published", "updated"]:
                val = getattr(e, key, None)
                if val:
                    try:
                        pub = datetime(*e.published_parsed[:6])
                    except Exception:
                        pub = None
                    break

            if pub is None:
                pub = datetime.utcnow()

            if pub < cutoff:
                continue

            rows.append({
                "PublishedAt": pub.isoformat(),
                "Ticker": ticker,
                "Region": region,
                "Headline": str(title)[:240],
                "Link": str(link)[:500],
                "Source": str(source)[:120] if source else "",
                "Query": q,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["Headline", "Link"]).reset_index(drop=True)
    return df
