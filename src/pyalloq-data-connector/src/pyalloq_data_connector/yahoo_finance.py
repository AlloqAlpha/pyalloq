import yfinance as yf
import pandas as pd
import datetime as dt
from pyalloq_data_connector.base import BaseDataClient
from pyalloq_core.data import MarketData


class YahooFinanceClient(BaseDataClient):
    """
    Adapter for Yahoo Finance.
    Does not require an API key.
    """

    def __init__(self):
        super().__init__(api_key=None)

    def fetch_raw_data(
        self, tickers: list[str], start: dt.datetime, end: dt.datetime
    ) -> dict[str, pd.DataFrame]:
        raw_dict = {}
        for t in tickers:
            yt = yf.Ticker(t)
            df = yt.history(start=start, end=end, auto_adjust=True)

            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                raw_dict[t] = df
            else:
                print(f"Warning: No data found for {t} on Yahoo Finance.")

        return raw_dict

    def fetch_historical_prices(
        self, tickers: list[str], start_date: dt.datetime, end_date: dt.datetime
    ) -> MarketData:
        return self.get_market_data(tickers, start_date, end_date)
