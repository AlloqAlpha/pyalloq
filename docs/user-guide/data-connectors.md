# Data Connectors

Data connectors fetch raw market data from external vendors and normalize it into a standardized `MarketData` object. All connectors inherit from `BaseDataConnector`.

---

## Yahoo Finance (`YahooFinanceClient`)

Free, no API key required. Suitable for research and prototyping.

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient

client = YahooFinanceClient()
data = client.get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start="2020-01-01",
    end="2024-01-01",
)
```

---

## Alpha Vantage (`AlphaVantageClient`)

Requires a free or premium API key from [alphavantage.co](https://www.alphavantage.co/).

```python
from pyalloq_data_connector.alpha_vantage import AlphaVantageClient

client = AlphaVantageClient(api_key="YOUR_API_KEY")
data = client.get_market_data(
    tickers=["AAPL", "MSFT"],
    start="2022-01-01",
    end="2024-01-01",
)
```

---

## Finnhub (`FinnhubClient`)

Real-time and historical stock data. Requires an API key from [finnhub.io](https://finnhub.io/).

```python
from pyalloq_data_connector.finnhub import FinnhubClient

client = FinnhubClient(api_key="YOUR_API_KEY")
data = client.get_market_data(
    tickers=["AAPL", "MSFT"],
    start="2023-01-01",
    end="2024-01-01",
)
```

---

## EOD Historical Data (`EODClient`)

Comprehensive global market data. Requires an API key from [eodhd.com](https://eodhd.com/).

```python
from pyalloq_data_connector.eod import EODClient

client = EODClient(api_key="YOUR_API_KEY")
data = client.get_market_data(
    tickers=["AAPL.US", "MSFT.US"],
    start="2020-01-01",
    end="2024-01-01",
)
```

---

## Adding Features to MarketData

After fetching prices, attach time-series features to your `MarketData`:

```python
from pyalloq_core.data import MarketData

# Fetch volume separately if your connector doesn't include it
volume_df = ...  # pd.DataFrame with same index as data.prices

enriched_data = MarketData(
    prices=data.prices,
    features={
        "volume": volume_df,
    },
    risk_free_rate=0.04,
)
```

---

## Implementing a Custom Connector

```python
from pyalloq_data_connector.base import BaseDataConnector
from pyalloq_core.data import MarketData
import pandas as pd

class MyDataConnector(BaseDataConnector):
    def get_market_data(
        self, tickers: list[str], start: str, end: str
    ) -> MarketData:
        # fetch from your data source
        prices = pd.DataFrame(...)
        return MarketData(prices=prices)
```
