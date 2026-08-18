# PyAlloq

A modern Python SDK and workspace for quantitative portfolio optimization, feature engineering, and zero-lookahead backtesting.

## Workspace Architecture

`pyalloq` is organized as a multi-package `uv` workspace:

- **[`pyalloq-core`](src/pyalloq-core)**: Core data abstractions (`MarketData`), pure interfaces (`BaseAllocator`, `BaseReturnEstimator`, `BaseCovarianceEstimator`), `StrategyPipeline`, and optimization results.
- **[`pyalloq-data-connector`](src/pyalloq-data-connector)**: Vendor-agnostic data adapters (Yahoo Finance, Alpha Vantage, Finnhub, EOD) standardizing raw payloads into aligned `MarketData`.
- **[`pyalloq-backtest`](src/pyalloq-backtest)**: Zero-lookahead historical simulation engines, cross-validation splitters, transaction cost models, and performance metrics.
- **[`pyalloq-features`](src/pyalloq-features)**: Feature engineering transformers, TA-Lib integration, scaling tools, and feature pipelines.
- **`pyalloq`**: Main SDK combining classical optimization (Markowitz, Risk Parity, Black-Litterman, HRP, NCO) and deep learning models.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/pyalloq.git
cd pyalloq

# Install all workspace packages and dependencies with uv
uv sync --all-extras

# Run type checking
uv run mypy src

# Run tests
uv run pytest

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## Quickstart

```python
import datetime as dt
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

# 1. Fetch Market Data
client = YahooFinanceClient()
data = client.get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start=dt.datetime(2023, 1, 1),
    end=dt.datetime(2024, 1, 1)
)

# 2. Compute Covariance & Allocate
cov_matrix = data.prices.pct_change().dropna().cov() * 252
allocator = RiskParityAllocator(tickers=data.assets)
result = allocator.allocate(data=data, cov_matrix=cov_matrix)

print("Optimal Weights:")
print(result.weights)
```

## License

MIT
