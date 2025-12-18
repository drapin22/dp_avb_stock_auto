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

# Artifacts
CALIBRATION_FILE = DATA_DIR / "calibration.json"
SCORES_FILE = DATA_DIR / "scores_stockd.csv"
MENTOR_OVERRIDES_FILE = DATA_DIR / "mentor_overrides.json"

FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"
PRICES_FILE = DATA_DIR / "prices_history.csv"

# OpenAI model names
MODEL_NAME = os.getenv("STOCKD_MODEL_NAME", "gpt-4.1-mini")
COACH_MODEL_NAME = os.getenv("STOCKD_COACH_MODEL_NAME", MODEL_NAME)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def has_telegram() -> bool:
    return bool(TELEGRAM_BOT_TOKEN.strip()) and bool(TELEGRAM_CHAT_ID.strip())
