from datetime import date
import pandas as pd
from stockd import settings


# Încarc parametrii din CSV (opțional)
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

    if len(df) < 21:
        return None

    df["ret"] = df["Close"].pct_change()

    last_close = df["Close"].iloc[-1]
    mom_5 = df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1.0
    mom_20 = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1.0
    vol_20 = df["ret"].iloc[-20:].std()

    return {
        "last_close": last_close,
        "mom_5": mom_5,
        "mom_20": mom_20,
        "vol_20": vol_20,
    }


def _score_from_features(feat: dict, horizon_days: int) -> float:
    """
    AICI e formula StockD adevărată.
    Deocamdată pun coeficienți exemplu – tu îi schimbi.
    """

    intercept = _PARAMS.get("intercept", 0.0)
    beta_m5 = _PARAMS.get("mom_5", 0.5)
    beta_m20 = _PARAMS.get("mom_20", 0.2)
    beta_vol = _PARAMS.get("vol_20", -0.3)

    raw = (
        intercept
        + beta_m5 * feat["mom_5"]
        + beta_m20 * feat["mom_20"]
        + beta_vol * feat["vol_20"]
    )

    scale = horizon_days / 5.0
    er_pct = raw * 100.0 * scale

    er_pct = max(min(er_pct, 20.0), -20.0)
    return er_pct


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Interfața oficială a modelului tău StockD.
    """

    result = holdings[["Ticker", "Region"]].copy()
    result["ER_Pct"] = 0.0

    for i, row in result.iterrows():
        ticker = row["Ticker"]

        feat = _compute_features_for_ticker(
            prices_history=prices_history,
            ticker=ticker,
            as_of=as_of,
        )

        if feat is None:
            er = 0.0
        else:
            er = _score_from_features(feat, horizon_days=horizon_days)

        result.at[i, "ER_Pct"] = er

    return result
