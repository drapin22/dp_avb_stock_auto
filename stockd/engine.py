from datetime import date
import pandas as pd
from stockd import settings  # ca să putem citi parametrii din data/, dacă vrei


# OPTIONAL: citim coeficienții dintr-un CSV, dacă vrei să nu-i pui hardcodați
# data/stockd_params.csv:
# feature,beta
# intercept,0.0
# mom_5,0.6
# mom_20,0.3
# vol_20,-0.2
try:
    _params_path = settings.DATA_DIR / "stockd_params.csv"
    _PARAMS = (
        pd.read_csv(_params_path)
        .set_index("feature")["beta"]
        .to_dict()
    )
except FileNotFoundError:
    _PARAMS = {}


def _compute_features_for_ticker(
    prices_history: pd.DataFrame,
    ticker: str,
    as_of: date,
) -> dict | None:
    """Calculează factorii de care are nevoie modelul pentru un singur ticker."""

    if prices_history.empty:
        return None

    df = prices_history[prices_history["Ticker"] == ticker].copy()
    if df.empty:
        return None

    df = df[df["Date"] <= pd.to_datetime(as_of)]
    df = df.sort_values("Date")

    # avem nevoie măcar de 21 de zile ca să facem mom_20
    if len(df) < 21:
        return None

    df["ret"] = df["Close"].pct_change()

    last_close = df["Close"].iloc[-1]
    # 5-day momentum
    mom_5 = df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1.0
    # 20-day momentum
    mom_20 = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1.0
    # 20-day volatility (σ)
    vol_20 = df["ret"].iloc[-20:].std()

    return {
        "last_close": last_close,
        "mom_5": mom_5,
        "mom_20": mom_20,
        "vol_20": vol_20,
    }


def _score_from_features(feat: dict, horizon_days: int) -> float:
    """
    AICI e formula StockD.
    raw = α + β1 * mom_5 + β2 * mom_20 + β3 * vol_20 + ...
    """

    # Poți seta coeficienții direct aici
    intercept = _PARAMS.get("intercept", 0.0)

    beta_m5 = _PARAMS.get("mom_5", 0.5)    # ← înlocuiești cu beta din calibrare
    beta_m20 = _PARAMS.get("mom_20", 0.2)  # ← la fel
    beta_vol = _PARAMS.get("vol_20", -0.3)

    raw = (
        intercept
        + beta_m5 * feat["mom_5"]
        + beta_m20 * feat["mom_20"]
        + beta_vol * feat["vol_20"]
    )

    # Presupunem că raw este pe 5 zile; dacă e definit altfel, ajustezi
    scale = horizon_days / 5.0
    er_pct = raw * 100.0 * scale  # în %

    # Control de amplitudine, ca să nu-ți sară în aer semnalele
    er_pct = max(min(er_pct, 20.0), -20.0)

    return er_pct
