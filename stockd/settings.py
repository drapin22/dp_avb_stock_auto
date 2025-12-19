from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PRICES_HISTORY = DATA_DIR / "prices_history.csv"
FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"

HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

MODEL_NAME = os.getenv("STOCKD_OPENAI_MODEL", "gpt-4.1-mini")
MODEL_VERSION_TAG = os.getenv("STOCKD_MODEL_VERSION", "StockD_V11.0_MKT_NEWS_MACRO")
FORECAST_NOTES = "weekly auto-forecast with macro/news/regime + learning"

MODEL_STATE_JSON = DATA_DIR / "model_state.json"
MENTOR_OVERRIDES_JSON = DATA_DIR / "mentor_overrides.json"

# telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_MAX_CHARS = int(os.getenv("TELEGRAM_MAX_CHARS", "3500"))

# openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# feature toggles
ENABLE_LLM_NEWS_ADJ = os.getenv("STOCKD_ENABLE_LLM_NEWS_ADJ", "1").strip() == "1"
ENABLE_MENTOR = os.getenv("STOCKD_ENABLE_MENTOR", "1").strip() == "1"
ENABLE_MACRO = os.getenv("STOCKD_ENABLE_MACRO", "1").strip() == "1"

# safety caps
MAX_ABS_ER_PCT = float(os.getenv("STOCKD_MAX_ABS_ER_PCT", "10.0"))
MAX_NEWS_DELTA_PP = float(os.getenv("STOCKD_MAX_NEWS_DELTA_PP", "1.5"))
