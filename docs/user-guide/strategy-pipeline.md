# Strategy Pipeline

`StrategyPipeline` is the **orchestrator** that connects your return estimator, covariance estimator, and allocator into a single callable unit. It is the primary object consumed by `WalkForwardEngine`.

---

## Basic Usage

```python
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

pipeline = StrategyPipeline(
    allocator=RiskParityAllocator(tickers=data.assets)
)
```

With defaults, `StrategyPipeline` uses:

- **Return estimator:** `EWMAReturnEstimator` (60-day span)
- **Covariance estimator:** `EmpiricalCovariance`

---

## Full Configuration

```python
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
from pyalloq.estimators.returns.classical.james_stein import JamesSteinReturnEstimator
from pyalloq.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq_core.enums import ObjectiveFunction

pipeline = StrategyPipeline(
    allocator=MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_SHARPE,
    ),
    returns_estimator=JamesSteinReturnEstimator(),
    cov_estimator=LedoitWolfShrinkage(),
    allocator_kwargs={},  # optional extra kwargs forwarded to allocator.allocate()
)
```

---

## How It Works

When `generate_weights(data)` is called — either directly or by the backtest engine — it runs the following sequence:

```python
def generate_weights(self, data: MarketData) -> pd.Series:
    expected_returns = self.returns_estimator.estimate(data)  # → pd.Series (N,)
    cov_matrix       = self.cov_estimator.estimate(data)      # → pd.DataFrame (N×N)

    result = self.allocator.allocate(
        data,
        cov_matrix=cov_matrix,
        expected_returns=expected_returns,
        **self.allocator_kwargs,
    )

    return result.weights  # → pd.Series (N,)
```

This means the pipeline is fully self-contained: given a `MarketData` slice, it produces portfolio weights.

---

## Using with WalkForwardEngine

The pipeline is the only thing `WalkForwardEngine` needs to know about your strategy:

```python
from pyalloq_backtest.engine import WalkForwardEngine

engine = WalkForwardEngine(pipeline=pipeline, rebalance_freq="ME")
results = engine.run(data)
```

The engine handles windowing, lookahead prevention, and cost accounting. The pipeline handles estimation and optimization.

---

## Calling Directly (Without Backtest)

You can also call the pipeline directly to get weights for a given data window, useful for live portfolio construction:

```python
# Use the last 252 days
window_data = data.slice_time(end_date=data.prices.index[-1], lookback=252)
weights = pipeline.generate_weights(window_data)
print(weights)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allocator` | `BaseAllocator` | required | The portfolio optimizer |
| `returns_estimator` | `BaseReturnEstimator \| None` | `EWMAReturnEstimator()` | Return estimator. Set to `None` to use default |
| `cov_estimator` | `BaseCovarianceEstimator \| None` | `EmpiricalCovariance()` | Covariance estimator. Set to `None` to use default |
| `allocator_kwargs` | `dict` | `{}` | Extra keyword arguments forwarded to `allocator.allocate()` |
