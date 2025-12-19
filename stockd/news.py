from __future__ import annotations

import re
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


# alias minimal; îl extinzi în timp automat (mentor poate propune update-uri)
ALIASES: Dict[Tuple[str, str], List[str]] = {
    ("PATH", "US"): ["UiPath", "UiPath Inc", "UiPath automation"],
    ("EL", "RO"): ["Electrica", "Electrica SA"],
    ("SNP", "RO"): ["OMV Petrom", "Petrom"],
    ("TLV", "RO"): ["Banca Transilvania", "BT Transilvania"],
}

# tickere cu risc de confuzie (cuvinte comune)
AMBIGUOUS_TICKERS = {"PATH", "EL", "AG"}


def _rss_url(query: str, locale: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={locale}&gl={gl}&ceid={ceid}"


def _parse_titles(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    out = []
    for item in soup.find_all("item"):
        t = item.find("title")
        if t and t.text:
            out.append(t.text.strip())
    return out


def _build_query(ticker: str, region: str) -> Tuple[str, List[str]]:
    ticker = ticker.upper().strip()
    region = region.upper().strip()

    aliases = ALIASES.get((ticker, region), [])

    # dacă nu avem alias, dar ticker e ambiguu, construim hint conservator
    if not aliases and ticker in AMBIGUOUS_TICKERS:
        if ticker == "PATH":
            aliases = ["UiPath"]
        elif ticker == "EL" and region == "RO":
            aliases = ["Electrica"]
        else:
            aliases = [ticker]

    if aliases:
        q = "(" + " OR ".join([f'"{a}"' for a in aliases]) + ")"
        return q, aliases

    # fallback: ticker simplu + regiune
    extra = "BVB" if region == "RO" else ""
    q = " ".join([ticker, extra]).strip()
    return q, []


def fetch_headlines_for_ticker(ticker: str, region: str, max_items: int = 8) -> List[str]:
    q, aliases = _build_query(ticker, region)

    try:
        url = _rss_url(q)
        r = requests.get(url, timeout=25)
        if not r.ok:
            return []
        titles = _parse_titles(r.text)
    except Exception:
        return []

    # dacă avem alias, filtrăm să nu prindă “path forward” etc
    if aliases:
        pat = re.compile("|".join([re.escape(a.lower()) for a in aliases]), re.IGNORECASE)
        titles = [t for t in titles if pat.search(t)]

    # dedup + limit
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_items:
            break
    return out
