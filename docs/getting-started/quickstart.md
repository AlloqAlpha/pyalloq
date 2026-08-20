# Quickstart

This guide walks you through building and backtesting a complete portfolio strategy end-to-end.

---

## Step 1 — Fetch Market Data

Use a data connector to download prices. All connectors return a standardized `MarketData` object.

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient

client = YahooFinanceClient()
data = client.get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    start="2020-01-01",
    end="2024-01-01",
)

print(data.assets)   # ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
print(data.prices.shape)  # (1006, 5)
```

---

## Step 2 — Choose an Allocator

Pick any allocator from `pyalloq.optimizers`. Every allocator implements the same `BaseAllocator` interface.

```python
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

allocator = RiskParityAllocator(tickers=data.assets)
```

---

## Step 3 — Build a Strategy Pipeline

`StrategyPipeline` wires together your **return estimator**, **covariance estimator**, and **allocator**. Defaults are EWMA returns and empirical covariance.

```python
from pyalloq_core.pipeline import StrategyPipeline

pipeline = StrategyPipeline(allocator=allocator)
```

To customize the estimators:

```python
from pyalloq.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.estimators.returns.classical.james_stein import JamesSteinReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=JamesSteinReturnEstimator(),
    cov_estimator=LedoitWolfShrinkage(),
)
```

---

## Step 4 — Run a Walk-Forward Backtest

`WalkForwardEngine` runs your pipeline on rolling windows of historical data, rebalancing at a specified frequency, and accounts for transaction costs.

```python
from pyalloq_backtest.engine import WalkForwardEngine

engine = WalkForwardEngine(
    pipeline=pipeline,
    rebalance_freq="ME",  # Monthly rebalancing
)

results = engine.run(data)
```

---

## Step 5 — Inspect Results

The engine returns a dictionary with portfolio returns, daily weights, and a performance tear sheet.

```python
print(results["tear_sheet"])
```

```
                       Value
total_return            1.23
annualized_return       0.21
annualized_volatility   0.14
sharpe_ratio            1.45
maximum_drawdown       -0.18
calmar_ratio            1.17
```

You can also access the raw returns and weight history:

```python
portfolio_returns = results["returns"]     # pd.Series
weight_history    = results["weights"]     # pd.DataFrame
```

---

## Complete Example

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator
from pyalloq.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine

# 1. Fetch data
data = YahooFinanceClient().get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    start="2020-01-01",
    end="2024-01-01",
)

# 2. Build pipeline
pipeline = StrategyPipeline(
    allocator=RiskParityAllocator(tickers=data.assets),
    cov_estimator=LedoitWolfShrinkage(),
)

# 3. Run backtest
results = WalkForwardEngine(pipeline=pipeline, rebalance_freq="ME").run(data)

# 4. Review performance
print(results["tear_sheet"])
```

---

## Next Steps

- [Concepts: Building Blocks](../concepts/building-blocks.md) — Understand the full architecture
- [Swapping Components](../concepts/swapping-components.md) — Experiment with different strategies
- [All Allocators](../user-guide/allocators.md) — See every available allocator
