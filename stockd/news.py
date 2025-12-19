from __future__ import annotations

import re
from typing import List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

ALIASES = {
    ("PATH", "US"): ["UiPath", "UiPath Inc", "UiPath stock"],
    ("BABA", "US"): ["Alibaba", "Alibaba Group", "Alibaba stock"],
    ("JD", "US"): ["JD.com", "JD.com stock"],
    ("NIO", "US"): ["NIO", "NIO Inc", "NIO stock"],
}


def _google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def _extract_rss_titles(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    titles = []
    for item in soup.find_all("item"):
        t = item.find("title")
        if t and t.text:
            titles.append(t.text.strip())
    return titles


def fetch_headlines_for_ticker(ticker: str, region: str, days: int = 10, max_items: int = 6) -> List[str]:
    ticker = str(ticker).upper().strip()
    region = str(region).upper().strip()

    alias_terms = ALIASES.get((ticker, region), [])
    base_terms = [ticker]
    if region == "RO":
        base_terms.append("BVB")

    if alias_terms:
        q = "(" + " OR ".join([f'"{a}"' for a in alias_terms]) + ")"
    else:
        q = " ".join([f'"{t}"' if " " in t else t for t in base_terms])

    url = _google_news_rss_url(q)

    try:
        r = requests.get(url, timeout=25)
        if not r.ok:
            return []
        titles = _extract_rss_titles(r.text)
    except Exception:
        return []

    if alias_terms:
        pat = re.compile("|".join([re.escape(a.lower()) for a in alias_terms]), re.IGNORECASE)
        titles = [t for t in titles if pat.search(t)]

    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_items:
            break

    return out
