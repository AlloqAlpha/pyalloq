# Swapping Components

Because every component in PyAlloq implements a clean interface, you can swap any piece of the stack with a single line of code — without changing anything else.

This page shows side-by-side examples of the most common swaps.

---

## Swapping Covariance Estimators

The covariance estimator has the biggest impact on optimizer stability, especially for Markowitz-family allocators. Try each with the same pipeline and compare results.

=== "Empirical (Default)"
    ```python
    from pyalloq.estimators.covariance.empirical import EmpiricalCovariance
    from pyalloq_core.pipeline import StrategyPipeline
    from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
    from pyalloq_core.enums import ObjectiveFunction

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
        cov_estimator=EmpiricalCovariance(),  # sample covariance — sensitive to outliers
    )
    ```

=== "Ledoit-Wolf Shrinkage"
    ```python
    from pyalloq.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
        cov_estimator=LedoitWolfShrinkage(),  # analytically optimal shrinkage intensity
    )
    ```

=== "RMT Denoising"
    ```python
    from pyalloq.estimators.covariance.random_matrix_theory import RandomMatrixTheoryEstimator

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
        cov_estimator=RandomMatrixTheoryEstimator(denoise=True, detone=False),
    )
    ```

=== "Semi-Covariance"
    ```python
    from pyalloq.estimators.covariance.semi_covariance import SemiCovariance

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
        cov_estimator=SemiCovariance(),  # penalizes downside volatility only
    )
    ```

---

## Swapping Return Estimators

For allocators that use expected returns (Max Sharpe, Max Return, Black-Litterman), the choice of return estimator directly impacts weights.

=== "EWMA (Default)"
    ```python
    from pyalloq.estimators.returns.classical.ewma import EWMAReturnEstimator

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MAX_SHARPE),
        returns_estimator=EWMAReturnEstimator(),  # exponentially weighted mean
    )
    ```

=== "James-Stein Shrinkage"
    ```python
    from pyalloq.estimators.returns.classical.james_stein import JamesSteinReturnEstimator

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MAX_SHARPE),
        returns_estimator=JamesSteinReturnEstimator(),  # shrinks towards grand mean
    )
    ```

=== "Momentum"
    ```python
    from pyalloq.estimators.returns.classical.momentum import CrossSectionalMomentumEstimator

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MAX_SHARPE),
        returns_estimator=CrossSectionalMomentumEstimator(),  # cross-sectional momentum signal
    )
    ```

=== "Implied Returns"
    ```python
    from pyalloq.estimators.returns.classical.implied import ImpliedReturnEstimator

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MAX_SHARPE),
        returns_estimator=ImpliedReturnEstimator(),  # reverse-optimization from market weights
    )
    ```

---

## Swapping Allocators

The allocator is completely decoupled from estimators and the backtest engine. Swap it freely.

=== "Risk Parity"
    ```python
    from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

    pipeline = StrategyPipeline(
        allocator=RiskParityAllocator(tickers=data.assets),
    )
    ```

=== "Markowitz (Min Vol)"
    ```python
    from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
    from pyalloq_core.enums import ObjectiveFunction

    pipeline = StrategyPipeline(
        allocator=MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
    )
    ```

=== "HRP (Hierarchical Risk Parity)"
    ```python
    from pyalloq.optimizers.machine_learning.hrp import HRPAllocator

    pipeline = StrategyPipeline(
        allocator=HRPAllocator(tickers=data.assets),
    )
    ```

=== "Max Diversification"
    ```python
    from pyalloq.optimizers.classical.max_diversification import MaxDiversificationAllocator

    pipeline = StrategyPipeline(
        allocator=MaxDiversificationAllocator(tickers=data.assets),
    )
    ```

=== "Equal Weight (Baseline)"
    ```python
    from pyalloq.optimizers.classical.naive import EqualWeightAllocator

    pipeline = StrategyPipeline(
        allocator=EqualWeightAllocator(tickers=data.assets),
    )
    ```

---

## Swapping Backtest Splitters

The splitter controls how the engine partitions history for each rebalance window. Swap it in `WalkForwardEngine`.

=== "Rolling Window (Default)"
    ```python
    from pyalloq_backtest.splitters import RollingWindowSplitter
    from pyalloq_backtest.engine import WalkForwardEngine

    engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=RollingWindowSplitter(lookback_window=252),  # fixed 1-year window
    )
    ```

=== "Expanding Window"
    ```python
    from pyalloq_backtest.splitters import ExpandingWindowSplitter

    engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=ExpandingWindowSplitter(min_periods=252),  # grows with all available history
    )
    ```

---

## Swapping Transaction Cost Models

=== "Flat Bps (Default)"
    ```python
    from pyalloq_backtest.costs import FlatBpsCostModel

    engine = WalkForwardEngine(
        pipeline=pipeline,
        cost_model=FlatBpsCostModel(bps=10.0),  # 10 basis points per trade
    )
    ```

=== "Almgren-Chriss Market Impact"
    ```python
    from pyalloq_backtest.costs import AlmgrenChrissCostModel

    # Requires data.features["volume"] to be set
    engine = WalkForwardEngine(
        pipeline=pipeline,
        cost_model=AlmgrenChrissCostModel(
            portfolio_aum=1_000_000,
            spread_bps=2.0,
            gamma=0.1,
        ),
    )
    ```

---

## Comparing Strategies Side-by-Side

Run multiple pipelines through the same engine and collect results for comparison:

```python
from pyalloq_backtest.engine import WalkForwardEngine

strategies = {
    "Equal Weight":   EqualWeightAllocator(tickers=data.assets),
    "Risk Parity":    RiskParityAllocator(tickers=data.assets),
    "HRP":            HRPAllocator(tickers=data.assets),
    "Min Volatility": MarkowitzAllocator(tickers=data.assets, objective=ObjectiveFunction.MIN_VOLATILITY),
}

results = {}
for name, allocator in strategies.items():
    pipeline = StrategyPipeline(allocator=allocator)
    engine = WalkForwardEngine(pipeline=pipeline, rebalance_freq="ME")
    results[name] = engine.run(data)

# Compare Sharpe ratios
for name, res in results.items():
    sharpe = res["tear_sheet"].loc["sharpe_ratio", "Value"]
    print(f"{name:<20} Sharpe: {sharpe:.2f}")
```
