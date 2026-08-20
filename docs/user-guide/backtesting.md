# Backtesting

`WalkForwardEngine` runs your `StrategyPipeline` across historical data, rebalancing at regular intervals, and accounts for transaction costs — with a zero-lookahead guarantee at every step.

---

## WalkForwardEngine

```python
from pyalloq_backtest.engine import WalkForwardEngine

engine = WalkForwardEngine(
    pipeline=pipeline,
    rebalance_freq="ME",                        # monthly rebalancing
    splitter=RollingWindowSplitter(252),         # 1-year rolling lookback
    cost_model=FlatBpsCostModel(bps=10.0),       # 10 bps per trade
)

results = engine.run(data)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `StrategyPipeline` | required | The strategy to backtest |
| `rebalance_freq` | `str` | `"ME"` | Pandas offset alias: `"ME"` (monthly), `"W"` (weekly), `"QE"` (quarterly) |
| `splitter` | `BaseWindowSplitter` | `RollingWindowSplitter(252)` | Controls how historical windows are sliced |
| `cost_model` | `BaseCostModel` | `FlatBpsCostModel(10.0)` | Transaction cost model applied at each rebalance |

### Returns

```python
results = engine.run(data)

results["returns"]     # pd.Series — daily portfolio returns (net of costs)
results["weights"]     # pd.DataFrame — daily forward-filled weights
results["tear_sheet"]  # pd.DataFrame — performance summary
```

---

## Window Splitters

The splitter controls how much historical data is visible to the pipeline at each rebalance date.

### Rolling Window (`RollingWindowSplitter`)

Uses a **fixed-length** lookback window. The window slides forward with each rebalance. Only rebalance dates with enough history (≥ `lookback_window` observations) are included.

```python
from pyalloq_backtest.splitters import RollingWindowSplitter

# Use exactly 1 year (252 trading days) of history at each rebalance
splitter = RollingWindowSplitter(lookback_window=252)
```

**When to use:** You want a consistent amount of data at each period, ignoring older observations. Appropriate for estimators sensitive to regime change.

---

### Expanding Window (`ExpandingWindowSplitter`)

Uses **all available history** up to each rebalance date, with a minimum threshold to start.

```python
from pyalloq_backtest.splitters import ExpandingWindowSplitter

# Start backtesting once 252 days of data are available; use all history thereafter
splitter = ExpandingWindowSplitter(min_periods=252)
```

**When to use:** When longer history improves estimator stability (e.g., Ledoit-Wolf, sample covariance), and you don't expect structural breaks.

---

### Purged K-Fold (`PurgedKFoldSplitter`)

Used for **ML model training and cross-validation**, not for backtesting. Splits data into K folds and applies an **embargo** (gap between train and test sets) to prevent information leakage from overlapping return windows.

```python
from pyalloq_backtest.splitters import PurgedKFoldSplitter

splitter = PurgedKFoldSplitter(n_splits=5, embargo_pct=0.01)

for train_data, test_data in splitter.split(data):
    model.fit(train_data)
    predictions = model.predict(test_data)
```

!!! warning "Not for standard backtesting"
    `PurgedKFoldSplitter` is designed for ML training loops, not for `WalkForwardEngine`. Use `RollingWindowSplitter` or `ExpandingWindowSplitter` for standard portfolio backtests.

---

## Transaction Cost Models

### Flat Bps (`FlatBpsCostModel`)

A flat percentage fee applied to the **absolute change in weights** at each rebalance. Suitable for retail or small-AUM use cases.

\[
\text{cost}_i = |\Delta w_i| \times \frac{\text{bps}}{10000}
\]

```python
from pyalloq_backtest.costs import FlatBpsCostModel

cost_model = FlatBpsCostModel(bps=10.0)  # 10 basis points per unit of turnover
```

---

### Almgren-Chriss (`AlmgrenChrissCostModel`)

Non-linear market impact cost model. Models both fixed bid-ask spread and non-linear price impact proportional to trade size relative to daily volume.

\[
\text{cost}_i = \left(\text{spread} + \gamma \sqrt{\frac{\text{trade\_size}_i}{\text{volume}_i}}\right) \times |\Delta w_i|
\]

```python
from pyalloq_backtest.costs import AlmgrenChrissCostModel

cost_model = AlmgrenChrissCostModel(
    portfolio_aum=1_000_000,  # portfolio size in USD
    spread_bps=2.0,            # bid-ask spread
    gamma=0.1,                 # market impact coefficient
)
```

!!! info "Requires volume data"
    `AlmgrenChrissCostModel` requires `data.features["volume"]` to be set on your `MarketData`. Volume is used to compute trade size as a fraction of daily market volume.

    ```python
    data = MarketData(
        prices=prices_df,
        features={"volume": volume_df},
    )
    ```

---

## Performance Metrics (TearSheet)

`MetricsTearSheet.generate(portfolio_returns)` produces a summary table of key performance metrics.

| Metric | Formula | Description |
|--------|---------|-------------|
| `total_return` | \((1+r)^{T} - 1\) | Cumulative return over full period |
| `annualized_return` | \((1 + \text{total\_return})^{1/Y} - 1\) | Geometric annualized return |
| `annualized_volatility` | \(\sigma_r \times \sqrt{252}\) | Annualized standard deviation |
| `sharpe_ratio` | \(\frac{r_a - r_f}{\sigma_a}\) | Risk-adjusted return |
| `maximum_drawdown` | \(\min_t \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}\) | Largest peak-to-trough decline |
| `calmar_ratio` | \(\frac{r_a}{|\text{MDD}|}\) | Return per unit of max drawdown |

```python
from pyalloq_backtest.metrics import MetricsTearSheet

tear_sheet = MetricsTearSheet.generate(
    portfolio_returns=results["returns"],
    risk_free_rate=0.04,
)
print(tear_sheet)
```

---

## Full Backtest Example

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator
from pyalloq.estimators.covariance.ledoit_wolf import LedoitWolfEstimator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import ExpandingWindowSplitter
from pyalloq_backtest.costs import AlmgrenChrissCostModel

data = YahooFinanceClient().get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start="2018-01-01", end="2024-01-01",
)

pipeline = StrategyPipeline(
    allocator=RiskParityAllocator(tickers=data.assets),
    cov_estimator=LedoitWolfEstimator(),
)

engine = WalkForwardEngine(
    pipeline=pipeline,
    rebalance_freq="ME",
    splitter=ExpandingWindowSplitter(min_periods=252),
    cost_model=FlatBpsCostModel(bps=5.0),
)

results = engine.run(data)
print(results["tear_sheet"])
```
