from datetime import date
import pandas as pd


def run_stockd_model(
    holdings: pd.DataFrame,
    prices_history: pd.DataFrame,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Interfața oficială a modelului tău StockD.

    Input:
      - holdings: DataFrame cu cel puțin coloanele ['Ticker', 'Region']
      - prices_history: toate prețurile istorice (RO + EU + US), dacă ai nevoie
      - as_of: data la care rulezi modelul (duminica)
      - horizon_days: orizontul pentru care vrei ER (ex: 5 zile)

    Output:
      - DataFrame cu:
          ['Ticker', 'Region', 'ER_Pct']
        unde ER_Pct e expected return % pe orizontul cerut.
    """

    # Copiem doar ce ne trebuie
    result = holdings[["Ticker", "Region"]].copy()
    result["ER_Pct"] = 0.0

    # TODO: aici vei pune logica REALĂ a modelului tău.
    # Ai acces la:
    #   - ticker, region
    #   - prices_history (factori, vol, etc.)
    #   - data 'as_of'
    #   - horizon_days

    for i, row in result.iterrows():
        ticker = row["Ticker"]
        region = row["Region"]

        # Placeholder temporar – îl înlocuim cu formula StockD
        er = 2.0  # <- AICI intră formula ta

        result.at[i, "ER_Pct"] = er

    return result
