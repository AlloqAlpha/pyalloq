# `pyalloq-viz` — Visualization & Tear Sheet Suite for PyAlloq

`pyalloq-viz` provides interactive Plotly visualizations, publication-ready charts, and self-contained HTML Tear Sheet dashboards for portfolio optimization, walk-forward backtesting, Monte Carlo simulations, and macroeconomic regime analysis.

---

## Overview & Design Philosophy

`pyalloq-viz` supports **two complementary ways** to visualize your quantitative workflows:

1. **Modular Plot Functions (`plot_*`)**:
   Every visualization is available as a standalone function that returns a styled `plotly.graph_objects.Figure` (supporting both `"alloq_dark"` and `"alloq_light"` institutional themes).
2. **Unified Tear Sheet Classes (`*TearSheet`)**:
   Four specialized tear sheet builders combine related charts and summary KPIs into either:
   - A multi-subplot interactive figure via `.to_figure()` / `.show()`
   - A self-contained, responsive HTML dashboard via `.to_html("report.html")`
   - A high-DPI static image via `.save_image("report.png")`

---

## Quickstart: End-to-End Workflow

```python
import numpy as np
import pandas as pd

from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq_simulation import (
    GMMRegimeGenerator,
    ViewBuilder,
    AbsoluteView,
    EntropyPooler,
)
import pyalloq_viz as viz

# 1. Prepare MarketData (with optional sector/hierarchy metadata)
dates = pd.bdate_range("2022-01-03", periods=500)
assets = ["AAPL", "MSFT", "TLT", "GLD"]
prices = pd.DataFrame(
    100.0 * np.cumprod(1.0 + np.random.normal(0.0005, 0.012, (500, 4)), axis=0),
    index=dates,
    columns=assets,
)
cross_sectional = pd.DataFrame(
    {"sector": ["Equities - Tech", "Equities - Tech", "Fixed Income", "Commodities"]},
    index=assets,
)
market_data = MarketData(prices=prices, cross_sectional=cross_sectional, risk_free_rate=0.02)

# 2. Run Walk-Forward Historical Backtest
cov_est = EmpiricalCovariance()
allocator = HRPAllocator(tickers=assets)
pipeline = StrategyPipeline(allocator=allocator, cov_estimator=cov_est)
bt_engine = WalkForwardEngine(
    pipeline=pipeline,
    splitter=RollingWindowSplitter(lookback_window=126),
    rebalance_freq="ME",
)
bt_result = bt_engine.run(market_data)

# 3. Fit GMM Regime Generator, Simulate Synthetic Futures & Apply Entropy Pooling Views
gmm_gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
gmm_gen.fit(market_data)
scenarios = gmm_gen.generate(n_paths=500, horizon=63)

builder = ViewBuilder(scenarios)
A_ineq, b_ineq, C_eq, d_eq = builder.build([
    AbsoluteView(asset="AAPL", operator=">=", target=0.02)
])
entropy_res = EntropyPooler(n_scenarios=scenarios.n_paths).fit(
    A_ineq=A_ineq, b_ineq=b_ineq, C_eq=C_eq, d_eq=d_eq
)

# 4. Generate Interactive HTML Tear Sheets!
viz.SimulationTearSheet(
    scenarios,
    entropy_result=entropy_res,
    ruin_threshold=-0.15,
    benchmark=bt_result,
).to_html("simulation_tearsheet.html")

viz.RobustnessTearSheet(bt_result).to_html("robustness_tearsheet.html")

viz.MacroRegimeTearSheet(
    entropy_result=entropy_res,
    equity_or_scenarios=bt_result,
    gmm_or_probs=gmm_gen,
    scenario_data=scenarios,
    market_data=market_data,
).to_html("macro_regime_tearsheet.html")

viz.AllocationTearSheet(
    backtest_result=bt_result,
    market_data=market_data,
    scenario_data=scenarios,
    allocator_or_pipeline=allocator,
).to_html("allocation_tearsheet.html")
```

---

## Detailed Guide by Tear Sheet

### 1. The Multiverse & Simulation Tear Sheet

Visualizes synthetic price/return paths (`ScenarioMarketData` or `MonteCarloResult`), optionally weighted by posterior probabilities from `EntropyResult`.

```python
from pyalloq_viz import (
    SimulationTearSheet,
    plot_fan_chart,
    plot_terminal_wealth_ruin,
    plot_cvar_tail_risk,
    plot_drawdown_fan_chart,
)

# Individual plots:
fig_fan = plot_fan_chart(
    scenarios,
    asset_weights=bt_result.weights.iloc[-1],
    entropy_result=entropy_res,          # Optional: weights quantiles by posterior x*
    percentiles=((5, 95), (10, 90), (25, 75)),
    benchmark=bt_result,                 # Optional: overlay realized equity curve
    theme="alloq_dark",
)

fig_ruin = plot_terminal_wealth_ruin(
    scenarios,
    entropy_result=entropy_res,
    ruin_threshold=-0.20,                # Highlights wealth dropping below 0.80 (-20%)
)

fig_cvar = plot_cvar_tail_risk(
    scenarios,                           # Accepts ScenarioMarketData, MonteCarloResult, or BacktestResult
    entropy_result=entropy_res,
    alpha=0.05,                          # 95% VaR and CVaR (Expected Shortfall) left-tail shading
)

fig_dd = plot_drawdown_fan_chart(
    scenarios,
    entropy_result=entropy_res,          # Underwater drawdown probability cones across all paths
)

# Combined Tear Sheet:
sim_ts = SimulationTearSheet(
    scenarios,
    asset_weights=bt_result.weights.iloc[-1],
    entropy_result=entropy_res,
    ruin_threshold=-0.20,
    cvar_alpha=0.05,
    benchmark=bt_result,
)
sim_ts.show()                            # Display 2x2 multi-subplot figure in Jupyter/IDE
sim_ts.to_html("multiverse_report.html") # Standalone HTML report with KPI summary cards
```

---

### 2. Robustness & Overfitting Tear Sheet

Integrates directly with `PBOEvaluator` and `CPCVSplitter` in `pyalloq_backtest` to test whether a strategy is genuinely robust or historically overfit.

#### Mode A: Single-Strategy Combinatorial Scrambling (CPCV) & Benchmark PBO
Recombines historical return blocks into scrambled chronological sequences of market regimes to test path-dependent survival:

```python
from pyalloq_backtest import CPCVSplitter, PBOEvaluator
from pyalloq_viz import RobustnessTearSheet, plot_pbo_distribution, plot_cpcv_paths

# Scramble historical returns across 6 regime blocks
splitter = CPCVSplitter(n_groups=6, k_test_groups=2, max_paths=120)
cpcv_res = splitter.scramble_timelines(bt_result)

# Plot scrambled equity paths & Sharpe vs. Max Drawdown scatter
fig_cpcv = plot_cpcv_paths(cpcv_res)

# Compare historical Sharpe against the scrambled/synthetic Sharpe distribution
fig_pbo = plot_pbo_distribution(bt_result, synthetic_comparison=cpcv_res)

# Or generate the full Robustness Tear Sheet in one line:
rob_ts = RobustnessTearSheet(bt_result, cpcv_or_synthetic=cpcv_res)
rob_ts.to_html("robustness_report.html")
```

#### Mode B: Multi-Strategy Combinatorial Symmetric Cross-Validation (CSCV)
Implements Bailey & Lopez de Prado's exact CSCV algorithm across a $(T \times M)$ matrix of $M$ candidate strategy configurations (or a list of `BacktestResult` objects):

```python
# trial_returns_df has shape (T_days, M_strategies)
pbo_eval = PBOEvaluator(n_partitions=8)
pbo_result = pbo_eval.evaluate_cscv(trial_returns_df)

# Plots Rank Logit (lambda) distribution (shaded for lambda <= 0) + IS vs. OOS Sharpe degradation
fig_cscv = plot_pbo_distribution(pbo_result)
```

---

### 3. Macro Regime & Entropy Tear Sheet

Visualizes subjective macroeconomic view adjustments (`EntropyPooler`) and Gaussian Mixture Model Markov regimes (`GMMRegimeGenerator`).

```python
from pyalloq_viz import (
    MacroRegimeTearSheet,
    plot_entropy_pooling,
    plot_regime_transitions,
)

# 1. Prior vs. Posterior Density (Entropy Pooling)
# Left panel: Uniform prior 1/M vs. sorted posterior x* with KL-divergence & Effective Scenarios (ENS)
# Right panel: Prior vs. Posterior view-shifted return distribution
fig_ep = plot_entropy_pooling(
    entropy_res,
    scenario_data=scenarios,
    asset_weights=bt_result.weights.iloc[-1],
)

# 2A. Historical Regime Transition Overlay
# Plots historical equity curve with background Bull/Bear regime shading + real-time p_k(t) stack
fig_reg_hist = plot_regime_transitions(
    bt_result,
    gmm_or_probs=gmm_gen,
    market_data=market_data,
)

# 2B. Synthetic Multiverse Regime Proportion Overlay
# Plots simulated equity fan cone + cross-sectional proportion of paths in each regime at step t
fig_reg_sim = plot_regime_transitions(
    scenarios,
    gmm_or_probs=gmm_gen,
)

# Combined Tear Sheet:
macro_ts = MacroRegimeTearSheet(
    entropy_result=entropy_res,
    equity_or_scenarios=bt_result,
    gmm_or_probs=gmm_gen,
    scenario_data=scenarios,
    market_data=market_data,
)
macro_ts.to_html("macro_regime_report.html")
```

---

### 4. Allocation Mechanics & Risk Tear Sheet

Inspects how optimizers allocate capital and risk across assets, how weights drift between rebalances, and how asset correlations and seasonal returns behave.

```python
from pyalloq.classical.optimizers.risk_budgeting import RiskBudgetingAllocator
from pyalloq_viz import (
    AllocationTearSheet,
    plot_multi_universe_frontier,
    plot_weight_drift_turnover,
    plot_hierarchical_composition,
    plot_risk_contributions,
    plot_correlation_and_periodic_returns,
)

cov_matrix = market_data.get_returns().cov() * 252.0
exp_returns = market_data.get_returns().mean() * 252.0

hrp_res = allocator.allocate(market_data, cov_matrix=cov_matrix)
rb_res = RiskBudgetingAllocator(tickers=assets).allocate(market_data, cov_matrix=cov_matrix)

# 1. Multi-Universe Efficient Frontier
# Renders Parametric Markowitz Frontier + Random Dirichlet Cloud + Optimal Allocations across synthetic paths
fig_ef = plot_multi_universe_frontier(
    expected_returns=exp_returns,
    cov_matrix=cov_matrix,
    scenario_data=scenarios,
    allocator_or_pipeline=allocator,
    optimal_portfolios=[hrp_res, rb_res],
)

# 2. Weight Drift & Rebalance Turnover Dynamics
# Reconstructs daily intra-period price drift (sawtooth pattern) and rebalance snap-backs + turnover bars
fig_drift = plot_weight_drift_turnover(bt_result, market_data=market_data)

# 3. Hierarchical Composition Treemap & Target vs. Current Comparison
# Resolves hierarchy from MarketData.cross_sectional, HRP/HERC/NCO clusters, or explicit dict
fig_tree = plot_hierarchical_composition(
    hrp_res,
    target_or_benchmark_weights=rb_res,
    market_data=market_data,
)

# 4. Capital Weights vs. Component Risk Contributions (PCRC) vs. Target Risk Budgets
fig_rc = plot_risk_contributions(rb_res, cov_matrix=cov_matrix)

# 5. Quasi-Diagonalized Asset Correlation Heatmap + Monthly Periodic Return Heatmap (Jan-Dec x Year)
fig_corr = plot_correlation_and_periodic_returns(
    market_data,
    portfolio_returns=bt_result,
    cluster_order=True,
)

# Combined Tear Sheet:
alloc_ts = AllocationTearSheet(
    optimization_result=hrp_res,
    backtest_result=bt_result,
    market_data=market_data,
    cov_matrix=cov_matrix,
    expected_returns=exp_returns,
    scenario_data=scenarios,
    allocator_or_pipeline=allocator,
    target_weights=rb_res,
)
alloc_ts.to_html("allocation_report.html")
```

---

## Themes & Styling (`alloq_dark` & `alloq_light`)

Every plotting function and TearSheet method accepts `theme="alloq_dark"` (default) or `theme="alloq_light"`:

```python
# Render light-mode publication chart
fig = plot_fan_chart(scenarios, theme="alloq_light")

# Export light-mode HTML dashboard
sim_ts.to_html("light_report.html", theme="alloq_light")
```
