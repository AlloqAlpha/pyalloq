import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyalloq_core.data import MarketData
from pyalloq_data_connector.alpha_vantage import AlphaVantageClient
from pyalloq_data_connector.base import BaseDataClient
from pyalloq_data_connector.eod import EODClient
from pyalloq_data_connector.finnhub import FinnhubClient
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient


class DummyClient(BaseDataClient):
    def fetch_raw_data(
        self, tickers: list[str], start_date: dt.datetime, end_date: dt.datetime
    ) -> dict[str, pd.DataFrame]:
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        out = {}
        for t in tickers:
            df = pd.DataFrame(
                {
                    "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "Open": [99.0, 100.0, 101.0, 102.0, 103.0],
                    "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                    "Low": [98.0, 99.0, 100.0, 101.0, 102.0],
                    "Volume": [1000, 2000, 1500, 1800, 2100],
                },
                index=dates,
            )
            out[t] = df
        return out


class TestBaseDataClient:
    def test_get_market_data_orchestrator(self) -> None:
        client = DummyClient()
        start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
        end = dt.datetime(2023, 1, 7, tzinfo=dt.UTC)
        tickers = ["AAPL", "MSFT"]

        md = client.get_market_data(tickers, start, end)
        assert isinstance(md, MarketData)
        assert list(md.prices.columns) == tickers
        assert len(md.prices) == 5
        assert "volume" in md.features
        assert "open" in md.features
        assert "high" in md.features
        assert "low" in md.features
        assert md.features["volume"].shape == (5, 2)


class TestYahooFinanceClient:
    @patch("yfinance.Ticker")
    def test_fetch_raw_data_mocked(self, mock_ticker_cls: MagicMock) -> None:
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        dummy_df = pd.DataFrame(
            {"Close": [150.0, 152.0, 151.0], "Volume": [1000, 1200, 1100]},
            index=dates,
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = dummy_df
        mock_ticker_cls.return_value = mock_instance

        client = YahooFinanceClient()
        start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
        end = dt.datetime(2023, 1, 5, tzinfo=dt.UTC)

        data = client.fetch_historical_prices(["AAPL"], start, end)
        assert isinstance(data, MarketData)
        assert "AAPL" in data.prices.columns
        assert len(data.prices) == 3


class TestAlphaVantageClient:
    def test_api_key_required(self) -> None:
        client = AlphaVantageClient(api_key=None)
        with pytest.raises(ValueError, match="Alpha Vantage requires an API key"):
            client.fetch_raw_data(
                ["AAPL"],
                dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2023, 1, 5, tzinfo=dt.UTC),
            )

    @patch("requests.get")
    def test_fetch_raw_data_mocked(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-03": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "99.0",
                    "4. close": "104.0",
                    "5. volume": "10000",
                },
                "2023-01-04": {
                    "1. open": "104.0",
                    "2. high": "107.0",
                    "3. low": "103.0",
                    "4. close": "106.0",
                    "5. volume": "12000",
                },
            }
        }
        mock_get.return_value = mock_response

        client = AlphaVantageClient(api_key="mock_key")
        raw = client.fetch_raw_data(
            ["AAPL"],
            dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2023, 1, 10, tzinfo=dt.UTC),
        )
        assert "AAPL" in raw
        df = raw["AAPL"]
        assert "close" in df.columns
        assert len(df) == 2
        assert df["close"].iloc[-1] == 106.0


class TestFinnhubClient:
    def test_api_key_required(self) -> None:
        client = FinnhubClient(api_key=None)
        with pytest.raises(ValueError, match="Finnhub requires an API key"):
            client.fetch_raw_data(
                ["AAPL"],
                dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2023, 1, 5, tzinfo=dt.UTC),
            )

    @patch("time.sleep", return_value=None)
    @patch("requests.get")
    def test_fetch_raw_data_mocked(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_response = MagicMock()
        # Finnhub timestamps: 1672704000 = 2023-01-03
        mock_response.json.return_value = {
            "s": "ok",
            "c": [150.0, 155.0],
            "h": [152.0, 156.0],
            "l": [149.0, 154.0],
            "o": [149.5, 154.5],
            "v": [50000, 60000],
            "t": [1672704000, 1672790400],
        }
        mock_get.return_value = mock_response

        client = FinnhubClient(api_key="test_token")
        raw = client.fetch_raw_data(
            ["AAPL"],
            dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2023, 1, 5, tzinfo=dt.UTC),
        )
        assert "AAPL" in raw
        df = raw["AAPL"]
        assert len(df) == 2
        assert "close" in df.columns
        assert df["close"].iloc[0] == 150.0


class TestEODClient:
    def test_api_key_required(self) -> None:
        client = EODClient(api_key=None)
        with pytest.raises(ValueError, match="EODHD requires an API key"):
            client.fetch_raw_data(
                ["AAPL"],
                dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2023, 1, 5, tzinfo=dt.UTC),
            )

    @patch("requests.get")
    def test_fetch_raw_data_mocked(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"date": "2023-01-03", "adjusted_close": 100.0, "volume": 1000},
            {"date": "2023-01-04", "adjusted_close": 102.0, "volume": 1500},
        ]
        mock_get.return_value = mock_response

        client = EODClient(api_key="mock_eod_key")
        raw = client.fetch_raw_data(
            ["AAPL"],
            dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2023, 1, 5, tzinfo=dt.UTC),
        )
        assert "AAPL" in raw
        df = raw["AAPL"]
        assert len(df) == 2
        assert "close" in df.columns
        assert df["close"].iloc[0] == 100.0
