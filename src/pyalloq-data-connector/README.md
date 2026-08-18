# pyalloq-data-connector

`pyalloq-data-connector` provides vendor-agnostic data adapters for **PyAlloq**. It handles fetching raw market data from various third-party APIs and automatically standardizes them into pure, time-aligned `MarketData` objects.

## Supported Adapters

- **`YahooFinanceClient`**: Free adapter using Yahoo Finance (`yfinance`). No API key required.
- **`AlphaVantageClient`**: Adapter for Alpha Vantage Time Series Daily API.
- **`FinnhubClient`**: Adapter for Finnhub's stock candle endpoint.
- **`EODHistoricalDataClient`**: Adapter for EOD Historical Data API.

## Features

- **Standardized Output**: Automatically converts heterogeneous JSON/DataFrame vendor payloads into `T x N` aligned price matrices (`MarketData`).
- **Missing Value Handling**: Implements forward filling (`ffill`) for prices and zero-filling for volume data.
- **Time Alignment**: Constructs unified datetime indices across all requested ticker symbols.

## Quick Example

```python
import datetime as dt
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient

client = YahooFinanceClient()
start = dt.datetime(2023, 1, 1)
end = dt.datetime(2024, 1, 1)

# Fetch standardized MarketData object
market_data = client.get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start=start,
    end=end
)

print(market_data.prices.head())
print(market_data.features["volume"].head())
```
