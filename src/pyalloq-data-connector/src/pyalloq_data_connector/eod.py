import requests
import pandas as pd
import datetime as dt
from pyalloq_data_connector.base import BaseDataClient


class EODClient(BaseDataClient):
    """Adapter for EOD Historical Data (eodhd.com)."""

    BASE_URL = "https://eodhd.com/api/eod"

    def fetch_raw_data(
        self, tickers: list[str], start: dt.datetime, end: dt.datetime
    ) -> dict[str, pd.DataFrame]:
        if not self.api_key:
            raise ValueError("EODHD requires an API key.")

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        raw_dict = {}
        for ticker in tickers:
            url = f"{self.BASE_URL}/{ticker}"
            params = {
                "api_token": self.api_key,
                "fmt": "json",
                "from": start_str,
                "to": end_str,
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(
                    f"Warning: EOD returned status {response.status_code} for {ticker}"
                )
                continue

            df = pd.DataFrame(response.json())
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df["close"] = df["adjusted_close"]
                raw_dict[ticker] = df

        return raw_dict
