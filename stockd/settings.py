from pathlib import Path

# Rădăcina repo-ului (dp_avb_stock_auto)
BASE_DIR = Path(__file__).resolve().parents[1]

# Directorul de date
DATA_DIR = BASE_DIR / "data"

# Fișierele de date
PRICES_HISTORY = DATA_DIR / "prices_history.csv"

HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

# Fișierul în care vrem să stea toate predicțiile săptămânale
FORECASTS_STOCKD = DATA_DIR / "forecasts_stockd.csv"
