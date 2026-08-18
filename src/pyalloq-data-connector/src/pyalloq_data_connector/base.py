from abc import ABC, abstractmethod
import pandas as pd
from pyalloq_core.data import MarketData
import datetime as dt


class BaseDataClient(ABC):
    """
    Abstract base class for all third-party data providers.
    Ensures raw API data is always converted to MarketData format.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @abstractmethod
    def fetch_raw_data(
        self,
        tickers: list[str],
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetches OHLCV data from the provider and packages it into MarketData.

        Args:
            tickers: List of asset symbols (e.g., ['AAPL', 'MSFT'])
            start_date: YYYY-MM-DD string
            end_date: YYYY-MM-DD string

        Returns:
            MarketData: The standardized PyAlloq data container.
        """
        pass

    def get_market_data(
        self,
        tickers: list[str],
        start: dt.datetime,
        end: dt.datetime,
    ) -> MarketData:
        """
        The orchestrator method. Standardizes whatever chaotic data the vendor
        returns into mathematically pure T x N matrices, perfectly aligned in time.
        """
        raw_dict = self.fetch_raw_data(tickers, start, end)

        closes, volumes, opens, highs, lows = {}, {}, {}, {}, {}
        all_dates = pd.DatetimeIndex([])

        for ticker, df in raw_dict.items():
            df.columns = [str(c).lower() for c in df.columns]

            closes[ticker] = df.get("close", df.get("adj close"))
            if "volume" in df.columns:
                volumes[ticker] = df["volume"]
            if "open" in df.columns:
                opens[ticker] = df["open"]
            if "high" in df.columns:
                highs[ticker] = df["high"]
            if "low" in df.columns:
                lows[ticker] = df["low"]

            dt_index = pd.to_datetime(df.index)
            all_dates = all_dates.union(dt_index)

        all_dates = all_dates.sort_values()

        def build_and_align(series_dict: dict, fill_method: str) -> pd.DataFrame:
            if not series_dict:
                return pd.DataFrame()

            df = pd.DataFrame(series_dict)
            df = df.reindex(all_dates)

            if fill_method == "ffill":
                return df.ffill()
            elif fill_method == "zero":
                return df.fillna(0.0)
            return df

        prices_df = build_and_align(closes, fill_method="ffill")

        features = {}
        if volumes:
            features["volume"] = build_and_align(volumes, fill_method="zero")
        if opens:
            features["open"] = build_and_align(opens, fill_method="ffill")
        if highs:
            features["high"] = build_and_align(highs, fill_method="ffill")
        if lows:
            features["low"] = build_and_align(lows, fill_method="ffill")

        valid_indices = prices_df.dropna(how="all").index
        prices_df = prices_df.loc[valid_indices]

        for key in features:
            features[key] = features[key].loc[valid_indices]

        return MarketData(prices=prices_df, features=features)
