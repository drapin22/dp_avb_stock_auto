# stockd/settings.py
from pathlib import Path

# Rădăcina repo-ului (dp_avb_stock_auto)
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

PRICES_HISTORY = DATA_DIR / "prices_history.csv"

HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

FORECASTS_FILE = DATA_DIR / "forecasts_stockd.csv"

# Meta info pentru model
MODEL_NAME = "gpt-4.1-mini"        # sau "gpt-4.1" dacă vrei ceva mai heavy
MODEL_VERSION_TAG = "StockD_V10.7F+"
FORECAST_NOTES = "weekly auto-forecast for next week"
