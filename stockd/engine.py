from datetime import date, timedelta

import pandas as pd


def _last_n_days_return(
    prices_history: pd.DataFrame,
    ticker: str,
    as_of: date,
    n: int = 20,
) -> float:
    """
    Calculează randamentul %-ual pe ultimele n zile de tranzacționare
    pentru un ticker, folosind coloanele:
      - Date (datetime / string)
      - Ticker
      - Close
    """
    if prices_history.empty:
        return 0.0

    df = prices_history.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # filtrăm doar tickerul respectiv și doar date <= as_of
    df = df[(df["Ticker"] == ticker) & (df["Date"] <= as_of)]
    df = df.sort_values("Date")

    if len(df) < 2:
        return 0.0

    # luăm ultimele n observații
    df = df.tail(n)
    first = df["Close"].iloc[0]
    last = df["Close"].iloc[-1]

    if first <= 0:
        return 0.0

    return (last / first - 1.0) * 100.0  # în %


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Interfața oficială a modelului StockD.

    Input:
      - holdings: DataFrame cu cel puțin coloanele ['Ticker', 'Region']
      - prices_history: toate prețurile istorice (RO + EU + US)
      - as_of: data la care rulezi modelul (de ex. vineri / duminică)
      - horizon_days: orizontul pentru care vrei ER (ex: 5 zile)

    Output:
      - DataFrame cu:
          ['Ticker', 'Region', 'ER_Pct']
        unde ER_Pct e expected return % pe orizontul cerut.
    """

    # Copiem coloanele esențiale
    result = holdings[["Ticker", "Region"]].copy()
    result["ER_Pct"] = 0.0

    # ------------------------------------------------------------------
    # AICI ESTE LOGICA MODELULUI
    # Momentan: o formulă simplă bazată pe momentum 20d,
    # scalată la orizontul de 5 zile.
    # TU poți înlocui bucata asta cu formula ta V10.7F+.
    # ------------------------------------------------------------------

    base_horizon = 5  # considerăm că V10.7F+ e pe 5 zile
    scale = horizon_days / base_horizon if base_horizon > 0 else 1.0

    for i, row in result.iterrows():
        ticker = row["Ticker"]
        # poți folosi și row["Region"] dacă ai logici diferite pe RO / EU / US

        # exemplu: momentum pe 20 de zile
        m20 = _last_n_days_return(
            prices_history=prices_history,
            ticker=ticker,
            as_of=as_of,
            n=20,
        )

        # FOARTE SIMPLIFICAT: ER = 0.5 * momentum_20d, scalat pe orizont
        er = 0.5 * m20 * scale

        # aici poți adăuga și alte semnale:
        # - volatilitate
        # - factor value / quality
        # - ajustare pe bază de risc etc.

        result.at[i, "ER_Pct"] = float(er)

    return result
