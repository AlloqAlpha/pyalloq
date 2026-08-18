# pyalloq-core

`pyalloq-core` provides the foundational data structures, pure interfaces, and core abstractions for the **PyAlloq** quantitative portfolio optimization SDK.

## Key Modules & Abstractions

- **`MarketData`**: Standardized parameter object holding aligned price time-series (`pd.DataFrame`), optional asset features (`dict[str, pd.DataFrame]`), cross-sectional data, asset lists, risk-free rates, and risk aversion parameters. Includes zero-lookahead time slicing (`slice_time`).
- **`BaseAllocator`**: Abstract base class for all portfolio allocation engines (Markowitz, Risk Parity, HRP, NCO, Deep Learning allocators, etc.).
- **`BaseReturnEstimator`**: Abstract base class for expected return estimators (Classical, EWMA, Black-Litterman, Factor models, Deep Learning).
- **`BaseCovarianceEstimator`**: Abstract base class for covariance matrix estimators (Empirical, EWMA, Ledoit-Wolf, Semi-covariance, RMT).
- **`StrategyPipeline`**: Pipeline orchestrator linking return estimators, covariance estimators, and portfolio allocators into an end-to-end strategy execution object.
- **`OptimizationResult`**: Standardized result container storing optimized portfolio weights, solver status, and metadata.

## Installation

```bash
uv add pyalloq-core
# Or inside workspace
uv sync
```

## Quick Example

```python
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction

# Create MarketData container
prices = pd.DataFrame(
    {"AAPL": [150.0, 152.5, 151.0], "MSFT": [300.0, 305.0, 302.0]},
    index=pd.date_range("2024-01-01", periods=3)
)

data = MarketData(prices=prices, risk_free_rate=0.04)

print(data.assets)  # ['AAPL', 'MSFT']
print(data.prices.head())
```
