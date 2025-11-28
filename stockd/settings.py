from pathlib import Path

# Root folder of repo (where .git is)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

PRICES_HISTORY = DATA_DIR / "prices_history.csv"
HOLDINGS_RO = DATA_DIR / "holdings_ro.csv"
HOLDINGS_EU = DATA_DIR / "holdings_eu.csv"
HOLDINGS_US = DATA_DIR / "holdings_us.csv"

def debug_print():
    print("[SETTINGS] ROOT =", ROOT)
    print("[SETTINGS] DATA_DIR =", DATA_DIR)
    print("[SETTINGS] PRICES_HISTORY =", PRICES_HISTORY)
