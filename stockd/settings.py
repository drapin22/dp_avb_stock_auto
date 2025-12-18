# stockd/settings.py
from __future__ import annotations

import os
from pathlib import Path

# Repo root = folderul care conține "stockd/"
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Canonical file paths (NEW)
# ---------------------------
PRICES_FILE = DATA_DIR / "prices_history.csv"
FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"

HOLDINGS_RO_FILE = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU_FILE = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US_FILE = DATA_DIR / "holdings_us.csv"

CALIBRATION_FILE = DATA_DIR / "calibration.json"
SCORES_FILE = DATA_DIR / "scores_stockd.csv"
MENTOR_OVERRIDES_FILE = DATA_DIR / "mentor_overrides.json"

# ---------------------------
# Backwards-compatible names (OLD)
# Keep these to avoid AttributeError in existing modules.
# ---------------------------
PRICES_HISTORY = PRICES_FILE
PRICES_HISTORY_FILE = PRICES_FILE

FORECASTS_STOCKD = FORECASTS_FILE
FORECASTS_PATH = FORECASTS_FILE

HOLDINGS_RO = HOLDINGS_RO_FILE
HOLDINGS_EU = HOLDINGS_EU_FILE
HOLDINGS_US = HOLDINGS_US_FILE

CALIBRATION_JSON = CALIBRATION_FILE
SCORES_STOCKD = SCORES_FILE

# ---------------------------
# OpenAI configuration
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STOCKD_MODEL_NAME = os.getenv("STOCKD_MODEL_NAME", "gpt-4.1-mini")
STOCKD_COACH_MODEL_NAME = os.getenv("STOCKD_COACH_MODEL_NAME", STOCKD_MODEL_NAME)

# ---------------------------
# Telegram
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def has_telegram() -> bool:
    return bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)
