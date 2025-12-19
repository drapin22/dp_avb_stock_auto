from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("STOCKD_DATA_DIR", BASE_DIR / "data"))
REPORTS_DIR = Path(os.getenv("STOCKD_REPORTS_DIR", BASE_DIR / "reports"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PRICES_HISTORY = Path(os.getenv("STOCKD_PRICES_HISTORY", DATA_DIR / "prices_history.csv"))
FORECASTS_STOCKD = Path(os.getenv("STOCKD_FORECASTS", DATA_DIR / "forecasts_stockd.csv"))

HOLDINGS_RO = Path(os.getenv("STOCKD_HOLDINGS_RO", DATA_DIR / "holdings_ro.csv"))
HOLDINGS_EU = Path(os.getenv("STOCKD_HOLDINGS_EU", DATA_DIR / "holdings_eu.csv"))
HOLDINGS_US = Path(os.getenv("STOCKD_HOLDINGS_US", DATA_DIR / "holdings_us.csv"))

CALIBRATION_JSON = Path(os.getenv("STOCKD_CALIBRATION_JSON", DATA_DIR / "calibration.json"))
SCORES_CSV = Path(os.getenv("STOCKD_SCORES_CSV", DATA_DIR / "scores_stockd.csv"))
MENTOR_OVERRIDES_JSON = Path(os.getenv("STOCKD_MENTOR_OVERRIDES_JSON", DATA_DIR / "mentor_overrides.json"))

CALIBRATION_JSON_REPORTS = REPORTS_DIR / "calibration.json"
SCORES_CSV_REPORTS = REPORTS_DIR / "scores_stockd.csv"
MENTOR_OVERRIDES_JSON_REPORTS = REPORTS_DIR / "mentor_overrides.json"

MODEL_EVAL_DETAILED = REPORTS_DIR / "model_eval_detailed.csv"
MODEL_EVAL_SUMMARY = REPORTS_DIR / "model_eval_summary.csv"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

TELEGRAM_SEND_ALL_SIGNALS = os.getenv("TELEGRAM_SEND_ALL_SIGNALS", "1").strip() == "1"
TELEGRAM_MAX_CHARS = int(os.getenv("TELEGRAM_MAX_CHARS", "3500"))
