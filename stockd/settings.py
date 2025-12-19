from pathlib import Path
import os

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

PRICES_HISTORY = DATA_DIR / "prices_history.csv"
FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"

# Backward-compat aliases (older modules might reference these names)
FORECASTS_STOCKD = FORECASTS_FILE
PRICES_FILE = PRICES_HISTORY  # legacy alias sometimes used

# Core model artifacts
MODEL_STATE_JSON = DATA_DIR / "model_state.json"
MENTOR_OVERRIDES_JSON = DATA_DIR / "mentor_overrides.json"

# Learning artifacts
CALIBRATION_FILE = DATA_DIR / "calibration.json"
CALIBRATION_JSON = CALIBRATION_FILE

SCORES_FILE = DATA_DIR / "scores_stockd.csv"
SCORES_CSV = SCORES_FILE

MODEL_EVAL_DETAILED = REPORTS_DIR / "model_eval_detailed.csv"
MODEL_EVAL_SUMMARY = REPORTS_DIR / "model_eval_summary.csv"

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Telegram hard limit is 4096 chars; keep safety buffer
TELEGRAM_MAX_CHARS = int(os.getenv("TELEGRAM_MAX_CHARS", "3500"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Feature flags
ENABLE_LLM_NEWS_ADJ = os.getenv("ENABLE_LLM_NEWS_ADJ", "1").strip() not in ("0", "false", "False", "")
ENABLE_LLM_POSTMORTEM = os.getenv("ENABLE_LLM_POSTMORTEM", "1").strip() not in ("0", "false", "False", "")

# Versioning
MODEL_VERSION_TAG = os.getenv("MODEL_VERSION_TAG", "StockD_V10.7F+")
FORECAST_NOTES = os.getenv("FORECAST_NOTES", "weekly auto-forecast for next week")
