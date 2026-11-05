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

### Combinatorial Purged Cross-Validation & Scrambling (`CPCVSplitter` & `PBOEvaluator`)

To guard against backtest overfitting and chronological regime luck, `pyalloq-backtest` includes **Combinatorial Purged Cross-Validation (`CPCVSplitter`)** and **Probability of Backtest Overfitting (`PBOEvaluator`)**:

1. **Regime Scrambling (`CPCVSplitter.scramble_timelines`)**: Partitions historical strategy returns into \(N\) chronological regime blocks and recombines them into scrambled timelines to test whether a strategy's Sharpe ratio and drawdowns survive when the sequence of market regimes is reordered.
2. **Probability of Backtest Overfitting (`PBOEvaluator`)**:
   - `evaluate_cscv(strategy_returns_matrix)`: Runs Bailey & López de Prado's Combinatorial Symmetric Cross-Validation (CSCV) across a \((T \times M)\) matrix of \(M\) strategy trials to compute the exact probability that the In-Sample optimal strategy underperforms the median Out-of-Sample (\(\mathbb{P}[\lambda_c \le 0]\)).
   - `evaluate_benchmark(historical_result, cpcv_or_mc_result)`: Evaluates a single strategy's historical Sharpe against a distribution of CPCV or Monte Carlo paths.

```python
from pyalloq_backtest import CPCVSplitter, PBOEvaluator

# 1. Scramble historical backtest across 6 regime blocks
cpcv = CPCVSplitter(n_groups=6, k_test_groups=2, max_paths=100)
cpcv_result = cpcv.scramble_timelines(results)

# 2. Evaluate single-strategy survival vs. scrambled timelines
pbo_eval = PBOEvaluator(n_partitions=8)
pbo_bench = pbo_eval.evaluate_benchmark(results, cpcv_result)
print(f"Prob(OOS Sharpe < 0): {pbo_bench.prob_oos_loss:.1%}")
```

!!! tip "Interactive Robustness Visualizations"
    Pass `results`, `cpcv_result`, or `pbo_result` directly to `RobustnessTearSheet` in [`pyalloq-viz`](visualization.md#2-robustness-overfitting-tear-sheet) to plot Rank Logit distributions, IS-vs-OOS Sharpe degradation, and scrambled equity cones.

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
| `probabilistic_sharpe_ratio` | \(\Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right)\) | Probability that Sharpe exceeds target \(SR^*\) |
| `deflated_sharpe_ratio` | \(\text{PSR}(\widehat{SR}_0)\) | Probability that Sharpe is statistically significant after penalizing for \(N\) trials, skewness, and kurtosis |
| `expected_max_sharpe_ratio` | \(\widehat{SR}_0(N, \text{Var}[\{\widehat{SR}_n\}])\) | Expected maximum Sharpe hurdle under the False Strategy Theorem |
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

## Multi-Strategy Comparison & Deflated Sharpe Ratio (`BenchmarkSuite`)

Evaluating multiple allocators simultaneously on the same historical trajectory introduces **Selection Bias** (the Multiple Testing Problem). `pyalloq-backtest` solves this via two integrated components:

1. **`BacktestTracker` (Singleton Ledger)**: Thread-safe singleton inside `WalkForwardEngine` that automatically increments the trial counter \(N\) and records each executed pipeline's Sharpe ratio, skewness, and kurtosis so cross-trial variance \(\text{Var}[\{\widehat{\text{SR}}_n\}]\) and the False Strategy hurdle \(\widehat{\text{SR}}_0\) are tracked automatically.
2. **`MultiStrategyComparator` (`BenchmarkSuite`)**: Runs a dictionary of `StrategyPipeline` (or `WalkForwardEngine`) instances concurrently over identical `MarketData` (and optional `ScenarioMarketData`), outputting a `ComparisonResult` with `.summary_table`, `.results`, `.scenario_results`, and `.attributions`.

```python
from pyalloq_backtest import BacktestTracker, BenchmarkSuite, RollingWindowSplitter, AlmgrenChrissCostModel
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq.classical import EqualWeightAllocator, RiskParityAllocator, MarkowitzAllocator, MultiPeriodMPCAllocator, LedoitWolfShrinkage
from pyalloq.ml import HRPAllocator

pipelines = {
    "EqualWeight": StrategyPipeline(allocator=EqualWeightAllocator(tickers=data.assets)),
    "RiskParity": StrategyPipeline(allocator=RiskParityAllocator(tickers=data.assets), cov_estimator=LedoitWolfShrinkage()),
    "HRP": StrategyPipeline(allocator=HRPAllocator(tickers=data.assets), cov_estimator=LedoitWolfShrinkage()),
    "Markowitz": StrategyPipeline(allocator=MarkowitzAllocator(tickers=data.assets), cov_estimator=LedoitWolfShrinkage()),
    "MultiPeriodMPC": StrategyPipeline(allocator=MultiPeriodMPCAllocator(tickers=data.assets), cov_estimator=LedoitWolfShrinkage()),
}

with BacktestTracker.session(reset=True) as tracker:
    suite = BenchmarkSuite(
        pipelines=pipelines,
        splitter=RollingWindowSplitter(lookback_window=252),
        cost_model=AlmgrenChrissCostModel(portfolio_aum=10_000_000, spread_bps=4.0, gamma=0.1),
        rebalance_freq="ME",
        benchmark="EqualWeight",
    )
    comp_result = suite.run(data)
    print(f"Total Trials Accounted (N): {tracker.n_trials}")
    print(comp_result.summary_table[["annualized_return", "sharpe_ratio", "deflated_sharpe_ratio", "maximum_drawdown"]])
```

---

## Brinson-Fachler, Factor & Frictional Drag Attribution (`PerformanceAttributor`)

To prove whether an allocator's outperformance stems from genuine skill, passive factor beta, or is eroded by execution mechanics, `PerformanceAttributor` decomposes active return \(R_p - R_b\) across three pillars:

1. **Classical Brinson-Fachler Decomposition (`compute_brinson_fachler`)**:
   - Decomposes active return into **Allocation Effect**, **Selection Effect**, and **Interaction Effect** across sectors or hierarchical asset clusters.
   - Supports **Carino (1999)** geometric multi-period linking (default) and arithmetic linking, with a 3-tier cluster resolution hierarchy (explicit `sector_map` \(\rightarrow\) `MarketData.cross_sectional["sector"]` \(\rightarrow\) automatic Ward/HRP correlation clustering).
2. **Factor-Implied Attribution (`compute_factor_attribution`)**:
   - Regresses active returns against systematic risk premia (`Market`, `Momentum`, `LowVolatility`, plus `Size` and `Value` when available, or custom `factor_returns`) using OLS with **Newey-West HAC** \(t\)-statistics.
3. **Frictional Drag Accounting (`compute_frictional_drag`, Ma & Smith, 2025)**:
   - Bridges **Theoretical Paper Return** (un-drifted target weights) \(\rightarrow\) **Alpha Decay & Drift Drag** \(\rightarrow\) **Gross Executed Return** \(\rightarrow\) **Proportional Spread/Commission Drag** \(\rightarrow\) **Almgren-Chriss Square-Root Market Impact Drag** \(\rightarrow\) **Short Borrow Drag** \(\rightarrow\) **Net Realized Return**.

```python
from pyalloq_backtest import PerformanceAttributor

attributor = PerformanceAttributor(linking_method="carino", hac_lags=5)
attr_res = attributor.attribute(
    backtest_result=comp_result.results["MultiPeriodMPC"],
    market_data=data,
    benchmark_weights=comp_result.results["EqualWeight"],
)

print(attr_res.brinson.sector_summary)        # Allocation, Selection, Interaction by sector/cluster
print(attr_res.factor.factor_summary)         # Factor exposures, Newey-West t-stats, annualized alpha
print(attr_res.frictional.waterfall_summary)  # Paper -> Drift/Decay -> Spread -> Market Impact -> Net
```

---

## Full Backtest Example

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
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
    cov_estimator=LedoitWolfShrinkage(),
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
