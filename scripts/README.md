# PyAlloq Example Scripts

This folder contains runnable end-to-end scripts demonstrating `pyalloq` workflows. All scripts include an automatic offline synthetic fallback if live market data APIs (e.g., Yahoo Finance) are unreachable.

## Available Scripts

1. **[`run_hrp_simulation_viz.py`](./run_hrp_simulation_viz.py)**
   - Runs Hierarchical Risk Parity (`HRPAllocator`) inside `WalkForwardEngine`.
   - Fits a `GMMRegimeGenerator` to simulate 500 multiverse futures and applies Meucci's `EntropyPooler` views.
   - Performs Combinatorial Purged Cross-Validation (`CPCVSplitter`) timeline scrambling.
   - Exports all 4 interactive HTML tear sheets to `visualizations_output/`.
   ```bash
   uv run python scripts/run_hrp_simulation_viz.py
   ```

2. **[`run_dl_e2e_allocation.py`](./run_dl_e2e_allocation.py)**
   - Builds rolling momentum and volatility features on `MarketData`.
   - Trains a `ConstrainedSpatioTemporalAttentionNet` (C-STAN) end-to-end neural allocator using `WalkForwardTrainer` and transaction-cost-aware `SharpeLoss`.
   - Evaluates `DeepAllocator` in `WalkForwardEngine` with intra-period weight drift, threshold rebalancing, and benchmark attribution against `EqualWeightAllocator` and `HRPAllocator`.
   - Exports interactive DL E2E HTML tear sheets to `visualizations_output/dl_e2e/`.
   ```bash
   uv run python scripts/run_dl_e2e_allocation.py
   ```

3. **[`run_multi_period_optimization.py`](./run_multi_period_optimization.py)**
   - Benchmarks all 3 Pillars of PyAlloq's Multi-Period Portfolio Optimization (MPO) architecture inside `WalkForwardEngine` under proportional transaction costs:
     - **Pillar 1**: `MultiPeriodMPCAllocator` (`backend="cvxpy"` with power cones & Gârleanu-Pedersen permanent impact, and `backend="admm"` operator splitting) paired with `AlphaDecayReturnEstimator`.
     - **Pillar 2**: `IPMOAllocator` with differentiable `MDFPOptimizationLayer` (implicit fixed-point backprop) trained via `HybridDFLLoss`.
     - **Pillar 3**: Two-Level `BayesianHRPAllocator` (RL-BHRP) and `RLAllocator` (`TrajectoryPolicyNet` trained via `DeepHedgingTrajectoryTrainer` + `WassersteinRegimeGate`).
   ```bash
   uv run python scripts/run_multi_period_optimization.py
   ```

4. **[`run_rl_multiverse_allocation.py`](./run_rl_multiverse_allocation.py)**
   - Fits a `GMMRegimeGenerator` to simulate a 3D multiverse (`ScenarioMarketData`) and attaches a `FeaturePipeline` (`RollingVolatility`).
   - Trains a continuous-control `SB3PortfolioAgent` (`PPO`) inside `POMDPPortfolioEnv` (`spaces.Dict` + `DictFlattenWrapper`) with `AlmgrenChrissCostModel` and Moody-Saffell `DifferentialSharpeReward`.
   - Trains a `HierarchicalMetaRLAllocator` (`MetaPortfolioEnv`) dynamically blending `MarkowitzAllocator`, `RiskParityAllocator`, `HRPAllocator`, and `MultiPeriodMPCAllocator`.
   - Evaluates all strategies in `WalkForwardEngine` and exports interactive HTML tear sheets to `visualizations_output/rl/`.
   ```bash
   uv run python scripts/run_rl_multiverse_allocation.py
   ```

5. **[`run_multi_strategy_benchmark.py`](./run_multi_strategy_benchmark.py)**
   - Runs `BenchmarkSuite` (`MultiStrategyComparator`) concurrently across `EqualWeightAllocator`, `RiskParityAllocator`, `HRPAllocator`, `MarkowitzAllocator`, and `MultiPeriodMPCAllocator` under `AlmgrenChrissCostModel`.
   - Tracks trial count $N$ and cross-trial Sharpe variance via the `BacktestTracker` singleton to compute the Deflated Sharpe Ratio (DSR, Bailey & López de Prado, 2014).
   - Executes `PerformanceAttributor` for Classical Brinson-Fachler sector/cluster decomposition, Factor-Implied regression, and Frictional Drag Accounting (Ma & Smith, 2025).
   - Exports `ComparisonTearSheet` to `visualizations_output/benchmark/`.
   ```bash
   uv run python scripts/run_multi_strategy_benchmark.py
   ```
