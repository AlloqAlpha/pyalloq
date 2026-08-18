import requests
import pandas as pd
import datetime as dt
from pyalloq_data_connector.base import BaseDataClient


class AlphaVantageClient(BaseDataClient):
    """Adapter for Alpha Vantage Time Series Daily."""

    BASE_URL = "https://www.alphavantage.co/query"

    def fetch_raw_data(
        self, tickers: list[str], start: dt.datetime, end: dt.datetime
    ) -> dict[str, pd.DataFrame]:
        if not self.api_key:
            raise ValueError("Alpha Vantage requires an API key.")

        raw_dict = {}
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        for ticker in tickers:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self.api_key,
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if "Time Series (Daily)" not in data:
                print(f"Warning: Alpha Vantage failed for {ticker}. Check API limits.")
                continue

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)

            df.columns = [col.split(" ")[1] for col in df.columns]
            df = df.astype(float)

            mask = (df.index >= start_ts) & (df.index <= end_ts)
            raw_dict[ticker] = df.loc[mask].sort_index()

        return raw_dict
