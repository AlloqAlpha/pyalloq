# Multi-Period Portfolio Optimization (MPO) — User Guide

PyAlloq provides a complete, modular suite of **Multi-Period Portfolio Optimization (MPO)** estimators, allocators, differentiable neural layers, and reinforcement learning environments that seamlessly plug into `StrategyPipeline` and `WalkForwardEngine`.

---

## How Stateful Rebalancing Works in `WalkForwardEngine`

In multi-period optimization, the optimal trade at date \(t\) depends on your **current drifted holdings** \(w_{t^-}\) immediately prior to rebalancing.

- `WalkForwardEngine` automatically tracks intra-period asset price drift and passes `current_weights=w_{t^-}` into `StrategyPipeline.generate_weights(..., current_weights=current_w_series)` on every rebalance date.
- Single-period allocators ignore `current_weights`, while multi-period allocators use `current_weights` to penalize turnover, model market impact, and return a [`MultiPeriodOptimizationResult`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq-core/src/pyalloq_core/results.py) containing both the immediate trade execution `result.weights` and the planned \(H\)-stage trajectory `result.weights_trajectory`.

---

## Pillar 1: Classical Convex MPC, ADMM & Stochastic MPC

### 1. Multi-Horizon Alpha Decay + Boyd/Gârleanu-Pedersen MPC

Combine [`AlphaDecayReturnEstimator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/classical/estimators/returns/multi_horizon.py) with [`MultiPeriodMPCAllocator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/classical/optimizers/multi_period.py):

```python
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfCovariance
from pyalloq.classical.estimators.returns.multi_horizon import AlphaDecayReturnEstimator
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_core.pipeline import StrategyPipeline

# 1. Multi-horizon return estimator with fast (5d), medium (21d), and slow (63d) half-lives
return_estimator = AlphaDecayReturnEstimator(
    horizon=5,
    half_lives=(5.0, 21.0, 63.0),
)

# 2. Boyd (2017) + Garleanu-Pedersen (2013) Convex MPC Allocator
mpc_allocator = MultiPeriodMPCAllocator(
    horizon=5,
    risk_aversion=3.0,
    spread_cost_bps=10.0,       # p=1 proportional bid-ask spread (no-trade zone)
    impact_coeff=0.05,          # p=3/2 square-root market impact cone
    quad_cost_coeff=0.01,       # p=2 temporary quadratic cost
    permanent_impact_eta=0.02,  # Garleanu-Pedersen permanent impact
    permanent_impact_rho=0.3,   # Market resilience decay rate
    max_turnover_per_stage=0.40,
    backend="cvxpy",            # Use "admm" for fast operator splitting
)

# Inspect a single H-stage trajectory directly
res = mpc_allocator.allocate(
    data=market_data,
    cov_matrix=LedoitWolfCovariance().estimate(market_data),
    expected_returns=return_estimator.estimate_trajectory(market_data),
)
print("Immediate Execution Weights (t+0):\n", res.weights)
print("Planned 5-Stage Trajectory:\n", res.weights_trajectory)

# Or run a full stateful Walk-Forward Backtest
pipeline = StrategyPipeline(
    allocator=mpc_allocator,
    return_estimator=return_estimator,
    cov_estimator=LedoitWolfCovariance(),
)
engine = WalkForwardEngine(
    pipeline=pipeline,
    rebalance_freq="W-FRI",
    cost_model=FlatBpsCostModel(bps=10.0),
)
bt_result = engine.run(market_data)
```

### 2. Fast Operator Splitting (`backend="admm"`)

When running large hyperparameter sweeps or large asset universes without solver overhead, switch `backend="admm"`:

```python
admm_allocator = MultiPeriodMPCAllocator(
    horizon=5,
    risk_aversion=3.0,
    spread_cost_bps=10.0,
    quad_cost_coeff=0.02,
    backend="admm",
    admm_max_iter=200,
)
```

### 3. Scenario-Tree Stochastic MPC with CVaR Tail Control

Optimize over Monte Carlo or Entropy-Pooled scenario tensors (`ScenarioMarketData`):

```python
from pyalloq.classical.optimizers.stochastic_mpc import StochasticScenarioMPCAllocator
from pyalloq_simulation.generators.gaussian import GaussianGenerator

gen = GaussianGenerator()
gen.fit(market_data)
scenarios = gen.generate(n_scenarios=250, horizon=5)

stoch_mpc = StochasticScenarioMPCAllocator(
    cvar_alpha=0.95,
    cvar_weight=2.0,
    spread_cost_bps=10.0,
)
stoch_res = stoch_mpc.allocate(scenarios)
```

---

## Pillar 2: Decision-Focused Learning (DFL) & IPMO

Use [`MultiHorizonPredictorNet`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/estimators/e2e/ipmo_allocator.py), [`MDFPOptimizationLayer`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/layers/mdfp.py), [`HybridDFLLoss`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/loss_functions/dfl.py), and [`IPMOAllocator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/estimators/e2e/ipmo_allocator.py) to train neural predictors end-to-end through the multi-period fixed-point optimization layer:

```python
import torch
from pyalloq.dl.estimators.e2e.ipmo_allocator import IPMOAllocator, MultiHorizonPredictorNet
from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer
from pyalloq.dl.loss_functions.dfl import HybridDFLLoss

n_assets = len(market_data.assets)
predictor = MultiHorizonPredictorNet(n_assets=n_assets, n_features=1, horizon=5)
mdfp_layer = MDFPOptimizationLayer(
    horizon=5,
    risk_aversion=2.5,
    cost_penalty=0.001,
    implicit_backward=True,  # O(1) memory implicit differentiation via Neumann VJPs
)
loss_fn = HybridDFLLoss(alpha=0.7, cost_penalty=0.001, risk_aversion=2.5)

# Wrap in IPMOAllocator for StrategyPipeline / WalkForwardEngine execution
ipmo_allocator = IPMOAllocator(
    predictor=predictor,
    mdfp_layer=mdfp_layer,
    lookback_window=30,
)
```

---

## Pillar 3: FinRL DRL, RL-BHRP, Deep Hedging & Thermodynamic Gates

### 1. Two-Level Bayesian HRP (`BayesianHRPAllocator`)

```python
from pyalloq.ml.optimizers.bhrp import BayesianHRPAllocator

bhrp = BayesianHRPAllocator(
    tau=0.3,
    return_tilt_strength=2.0,
    turnover_penalty=0.5,
)
```

### 2. Deep Hedging Trajectory Policy + Thermodynamic Regime Gate

```python
from pyalloq.rl import (
    DeepHedgingTrajectoryTrainer,
    RLAllocator,
    TrajectoryPolicyNet,
    WassersteinRegimeGate,
)

policy = TrajectoryPolicyNet(n_assets=len(market_data.assets), lookback_window=20)
trainer = DeepHedgingTrajectoryTrainer(
    policy=policy,
    horizon=5,
    transaction_cost_bps=10.0,
    risk_aversion=2.0,
)
trainer.fit_on_market_data(market_data, epochs=15)

regime_gate = WassersteinRegimeGate(fast_window=15, slow_window=45, w2_threshold=0.025)
rl_allocator = RLAllocator(
    policy=policy,
    lookback_window=20,
    regime_gate=regime_gate,
)
```

---

## Visualizing Multi-Period Optimization with `pyalloq-viz`

[`pyalloq-viz`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq-viz/src/pyalloq_viz/__init__.py) provides dedicated interactive Plotly charts and [`MultiPeriodTearSheet`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq-viz/src/pyalloq_viz/tearsheets/multi_period_ts.py) for multi-period trajectories, alpha decay surfaces, thermodynamic regime gates, and cross-pillar net-of-cost comparisons:

```python
from pyalloq_viz import (
    MultiPeriodTearSheet,
    plot_alpha_decay_surface,
    plot_mpc_trajectory,
    plot_mpo_pillar_comparison,
    plot_thermodynamic_regime_gate,
)

# 1. Individual interactive figures
fig_traj = plot_mpc_trajectory(res, current_weights=current_w)
fig_decay = plot_alpha_decay_surface(return_estimator.estimate_horizon(market_data))
fig_gate = plot_thermodynamic_regime_gate(market_data, fast_window=15, slow_window=45)
fig_bench = plot_mpo_pillar_comparison({"MPC": bt_result})

# 2. Standalone HTML Tear Sheet Dashboard
ts = MultiPeriodTearSheet(
    mpo_result=res,
    horizon_forecasts=return_estimator.estimate_horizon(market_data),
    market_data=market_data,
    strategy_comparison={"MPC": bt_result},
)
ts.to_html("visualizations_output/mpo/multi_period_tearsheet.html")
```
