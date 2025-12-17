# stockd/settings.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Prices
PRICES_HISTORY = DATA_DIR / "prices_history.csv"
PRICES_ALL_CSV = DATA_DIR / "prices_all.csv"  # folosit de weekly_report (dacă există)

# Holdings
HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

# Forecasts
FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"
FORECASTS_CSV = FORECASTS_FILE  # alias pentru weekly_report

# Learning outputs
MODEL_EVAL_DETAILED = REPORTS_DIR / "model_eval_detailed.csv"
MODEL_EVAL_SUMMARY = REPORTS_DIR / "model_eval_summary.csv"
CALIBRATION_FILE = DATA_DIR / "calibration.json"
SCORES_FILE = DATA_DIR / "scores_stockd.csv"
MENTOR_OVERRIDES_FILE = DATA_DIR / "mentor_overrides.json"

# Model meta
MODEL_NAME = "gpt-4.1-mini"
COACH_MODEL_NAME = "gpt-4.1-mini"  # mentor/coach
MODEL_VERSION_TAG = "StockD_V10.7F+"
FORECAST_NOTES = "weekly auto-forecast for next week"
