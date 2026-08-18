import requests
import pandas as pd
import datetime as dt
from pyalloq_data_connector.base import BaseDataClient
import time


class FinnhubClient(BaseDataClient):
    """Adapter for Finnhub's stock candle endpoint."""

    BASE_URL = "https://finnhub.io/api/v1/stock/candle"

    def fetch_raw_data(
        self, tickers: list[str], start: dt.datetime, end: dt.datetime
    ) -> dict[str, pd.DataFrame]:
        if not self.api_key:
            raise ValueError("Finnhub requires an API key.")

        raw_dict = {}

        start_unix = int(start.timestamp())
        end_unix = int(end.timestamp())

        for ticker in tickers:
            params: dict[str, str | int] = {
                "symbol": ticker,
                "resolution": "D",  # Daily bars
                "from": start_unix,
                "to": end_unix,
                "token": self.api_key,
            }

            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if data.get("s") != "ok":
                print(f"Warning: Finnhub returned no data or error for {ticker}")
                continue

            df = pd.DataFrame(
                {
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"],
                },
                index=pd.to_datetime(data["t"], unit="s"),
            )

            dt_idx = pd.DatetimeIndex(df.index)
            df.index = dt_idx.tz_localize(None).normalize()
            raw_dict[ticker] = df
            time.sleep(0.5)

        return raw_dict
