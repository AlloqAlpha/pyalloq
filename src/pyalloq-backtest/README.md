# pyalloq-backtest

`pyalloq-backtest` provides zero-lookahead historical simulation engines, rolling window splitters, transaction cost modeling, and quantitative performance metrics for **PyAlloq**.

## Key Modules

- **`BacktestEngine`**: Executes walk-forward strategy evaluation with periodic rebalancing, transaction cost deduction, and turnover tracking.
- **`splitters`**: Provides time-series train/test splitters and expanding/rolling window generators for cross-validation.
- **`costs`**: Transaction cost functions (linear bps costs, fixed fee per trade, slippage models).
- **`metrics`**: Comprehensive performance metrics including Sharpe ratio, Sortino ratio, Max Drawdown, CAGR, Annualized Volatility, and Turnover.

## Quick Example

```python
from pyalloq_backtest.engine import BacktestEngine
from pyalloq_backtest.costs import linear_cost

# Initialize zero-lookahead backtest engine
engine = BacktestEngine(
    rebalance_frequency=21,  # Monthly rebalancing (21 trading days)
    cost_model=linear_cost(bps=10.0)  # 10 bps transaction cost
)

# Run historical simulation on MarketData and StrategyPipeline
# results = engine.run(market_data, strategy_pipeline)
```
