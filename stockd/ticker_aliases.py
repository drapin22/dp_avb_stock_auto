# stockd/ticker_aliases.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import yfinance as yf

from stockd import settings


ALIASES_PATH = settings.DATA_DIR / "ticker_aliases.json"


DEFAULT_SPECIAL = {
    ("PATH", "US"): {"company": "UiPath", "query": "\"UiPath\" OR \"UiPath Inc\" OR (UiPath AND RPA)"},
    ("FP", "RO"): {"company": "Fondul Proprietatea", "query": "\"Fondul Proprietatea\" OR \"FP\" AND BVB"},
    ("TLV", "RO"): {"company": "Banca Transilvania", "query": "\"Banca Transilvania\" OR TLV BVB"},
}


def load_aliases() -> Dict[str, dict]:
    if ALIASES_PATH.exists():
        try:
            return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_aliases(d: Dict[str, dict]) -> None:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ALIASES_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def key(ticker: str, region: str) -> str:
    return f"{ticker}::{region}"


def build_query(ticker: str, region: str, aliases: Optional[Dict[str, dict]] = None) -> str:
    aliases = aliases or load_aliases()
    k = key(ticker, region)
    if k in aliases and isinstance(aliases[k], dict) and aliases[k].get("query"):
        return str(aliases[k]["query"])

    # fallback: ticker plus market tag
    if region == "US":
        return f"{ticker} stock OR {ticker} earnings OR {ticker} guidance"
    if region == "EU":
        return f"{ticker} OR {ticker} Aktie OR {ticker} results"
    return f"{ticker} BVB OR {ticker} rezultate OR {ticker} raport"


def auto_enrich_aliases(ticker: str, region: str, aliases: Dict[str, dict]) -> Dict[str, dict]:
    """
    Tries to add company name based query for US/EU tickers via yfinance.
    """
    k = key(ticker, region)
    if k in aliases and aliases[k].get("query"):
        return aliases

    # seed specials
    if (ticker, region) in DEFAULT_SPECIAL:
        aliases[k] = DEFAULT_SPECIAL[(ticker, region)]
        return aliases

    if region not in {"US", "EU"}:
        return aliases

    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or info.get("name")
        if name:
            # make a disambiguating query: company name dominates
            q = f"\"{name}\" OR ({ticker} AND stock)"
            aliases[k] = {"company": str(name), "query": q}
    except Exception:
        pass

    return aliases
