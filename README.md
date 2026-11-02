# PyAlloq

A modern Python SDK and workspace for quantitative portfolio optimization, feature engineering, and zero-lookahead backtesting.

## Workspace Architecture

`pyalloq` is organized as a multi-package `uv` workspace:

- **[`pyalloq-core`](src/pyalloq-core)**: Core data abstractions (`MarketData`, `ScenarioMarketData`), pure interfaces (`BaseAllocator`, `BaseReturnEstimator`, `BaseCovarianceEstimator`, `BaseScenarioGenerator`), `StrategyPipeline`, `ParallelMonteCarloEngine`, and optimization/simulation results.
- **[`pyalloq-data-connector`](src/pyalloq-data-connector)**: Vendor-agnostic data adapters (Yahoo Finance, Alpha Vantage, Finnhub, EOD) standardizing raw payloads into aligned `MarketData`.
- **[`pyalloq-backtest`](src/pyalloq-backtest)**: Zero-lookahead historical simulation engines, cross-validation splitters, transaction cost models, and performance metrics.
- **[`pyalloq-features`](src/pyalloq-features)**: Feature engineering transformers, TA-Lib integration, scaling tools, and feature pipelines.
- **[`pyalloq-simulation`](src/pyalloq-simulation)**: Institutional Monte Carlo scenario generators (`GaussianGenerator`, `PCAFactorGenerator`, `FHSGenerator`, `GARCHEVTCopulaGenerator`, `GMMRegimeGenerator`), `FactorBackcastImputer`, Meucci `EntropyPooler` (first- and second-moment views), and path-dependent `ScenarioEvaluator`.
- **[`pyalloq-viz`](src/pyalloq-viz)**: Interactive Plotly visualizations and standalone HTML tear sheets (`SimulationTearSheet`, `RobustnessTearSheet`, `MacroRegimeTearSheet`, `AllocationTearSheet`, `MultiPeriodTearSheet`) for portfolio optimization, multi-period trajectories, thermodynamic regime gating, PBO/CPCV robustness testing, entropy pooling, and regime analysis.
- **`pyalloq`**: Main SDK combining:
  - **Classical & Multi-Period Convex Optimization (`pyalloq.classical`)**: Markowitz, Risk Parity, CVaR, Black-Litterman, Multi-Horizon `AlphaDecayReturnEstimator`, `MultiPeriodMPCAllocator` (Boyd 2017 & Gârleanu-Pedersen 2013 with `cvxpy` power cones and high-speed `admm` operator splitting), and `StochasticScenarioMPCAllocator`.
  - **Hierarchical & Graph Machine Learning (`pyalloq.ml`)**: `HRPAllocator`, `NCOAllocator`, and Two-Level Bayesian `BayesianHRPAllocator` (RL-BHRP).
  - **Decision-Focused Deep Learning (`pyalloq.dl`)**: End-to-End `DeepAllocator` (C-STAN), `DeepCovarianceEstimator`, `MDFPOptimizationLayer` (Entropic Mirror-Descent Fixed-Point with \(O(1)\)-memory implicit differentiation), `SPOPlusLoss`, `HybridDFLLoss`, and `IPMOAllocator`.
  - **FinRL Deep Reinforcement Learning & Thermodynamic Gates (`pyalloq.rl`)**: `PortfolioGymEnv`, `ScenarioPortfolioEnv`, `TrajectoryPolicyNet`, `DeepHedgingTrajectoryTrainer`, `WassersteinRegimeGate` (Bures-Wasserstein \(W_2\) & Carnot efficiency bound), and `RLAllocator`.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/pyalloq.git
cd pyalloq

# Install all workspace packages and dependencies with uv
uv sync --all-extras

# Run type checking
uv run mypy src

# Run tests
uv run pytest

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## Quickstart

```python
import datetime as dt
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine

# 1. Fetch Market Data
client = YahooFinanceClient()
data = client.get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start=dt.datetime(2023, 1, 1),
    end=dt.datetime(2024, 1, 1)
)

# 2. Build Strategy Pipeline
allocator = RiskParityAllocator(tickers=data.assets)
pipeline = StrategyPipeline(allocator=allocator)

# 3. Run Walk-Forward Backtest
engine = WalkForwardEngine(pipeline=pipeline, rebalance_freq="ME")
results = engine.run(data)

# 4. Inspect Results & Performance Metrics
print("Performance Tear Sheet:")
print(results["tear_sheet"])
```

## License

MIT
