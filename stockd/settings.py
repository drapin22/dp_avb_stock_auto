# stockd/settings.py
from pathlib import Path

# Rădăcina repo-ului (dp_avb_stock_auto)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# FISIERE CANONICE
PRICES_HISTORY = DATA_DIR / "prices_history.csv"
FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"

# HOLDINGS
HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

# LEARNING OUTPUTS
MODEL_EVAL_DETAILED = DATA_DIR / "model_eval_detailed.csv"
MODEL_EVAL_SUMMARY = DATA_DIR / "model_eval_summary.csv"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

# ALIASURI pentru compatibilitate cu scripturi vechi
FORECASTS_CSV = FORECASTS_FILE
PRICES_ALL_CSV = PRICES_HISTORY

# MODEL META
MODEL_NAME = "gpt-4.1-mini"
MODEL_VERSION_TAG = "StockD_V10.7F+"
FORECAST_NOTES = "weekly auto-forecast for next week"
