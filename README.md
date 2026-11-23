# PyAlloq

A modern Python SDK and workspace for quantitative portfolio optimization, feature engineering, and zero-lookahead backtesting.

## Workspace Architecture

`pyalloq` is organized as a multi-package `uv` workspace:

- **[`pyalloq-core`](src/pyalloq-core)**: Core data abstractions (`MarketData`, `ScenarioMarketData`), pure interfaces (`BaseAllocator`, `BaseReturnEstimator`, `BaseCovarianceEstimator`, `BaseViewGenerator`, `BaseScenarioGenerator`, `BaseRegimeDetector`), `StrategyPipeline`, `ParallelMonteCarloEngine`, `DiscreteShareAllocator`, Conformal Risk (`ConformalPIDCalibrator`, `ConformalRiskController`), and online meta-weighting (`pyalloq_core.meta`).
- **[`pyalloq-data-connector`](src/pyalloq-data-connector)**: Vendor-agnostic `BaseDataClient` adapters (Yahoo Finance, Alpha Vantage, Finnhub, EOD) standardizing raw payloads into aligned `MarketData`.
- **[`pyalloq-backtest`](src/pyalloq-backtest)**: Zero-lookahead `WalkForwardEngine`, rolling/expanding/purged & `CPCVSplitter` / `PBOEvaluator`, 4 transaction cost models, `PerformanceAttributor`, `MetricsTearSheet` (with DSR/MinTRL/FDR), and `MultiStrategyComparator`.
- **[`pyalloq-features`](src/pyalloq-features)**: `FeaturePipeline`, technical & cross-sectional transformers, `RollingZScoreScaler`, Point-in-Time option-implied volatility/skew/kurtosis (`PITOptionsMFIVTransformer`), Rough Volatility & Lyndon Log-Signature transformers (`LiftedHestonRoughVolTransformer`, `LogSignatureTransformer`), and Statistical Jump Models (`SparseJumpModelDetector`, `ContinuousJumpModelDetector`).
- **[`pyalloq-simulation`](src/pyalloq-simulation)**: 8 Monte Carlo & generative scenario engines (`GaussianGenerator`, `PCAFactorGenerator`, `FHSGenerator`, `GARCHEVTCopulaGenerator`, `GMMRegimeGenerator`, `ReactionDiffusionGenerator`, `ConditionalDiffusionGenerator`, `NeuralSDEGenerator`), `FactorBackcastImputer`, Meucci `EntropyPooler` (first- and second-moment views), and path-dependent `ScenarioEvaluator`.
- **[`pyalloq-viz`](src/pyalloq-viz)**: Interactive Plotly visualizations and 6 standalone HTML tear sheets (`SimulationTearSheet`, `RobustnessTearSheet`, `MacroRegimeTearSheet`, `AllocationTearSheet`, `MultiPeriodTearSheet`, `ComparisonTearSheet`) including spectral contagion networks (`plot_spectral_contagion_network`, `plot_fragility_hawkes_regime_portrait`) and SetTransformer diagnostics (`plot_set_transformer_diagnostics`).
- **`pyalloq`**: Main SDK combining:
  - **Classical, Robust, Conformal & Multi-Period Optimization (`pyalloq.classical`)**: Markowitz (box bounds, market-neutral, leverage, turnover, sector constraints, efficient frontier), Risk Parity, Risk Budgeting, Max Diversification, `MeanCVaRAllocator` & `CDaRAllocator`, `WassersteinDROAllocator`, `CPPSAllocator`, `ConformalKellyOverlay`, `ConformalSafetyGateAllocator`, `EnsembleAllocator`, `RegimeSwitchingAllocator`, Black-Litterman (`BlackLittermanReturnEstimator`, `BlackLittermanCovarianceEstimator`, `AssetSpecificJumpViewGenerator`), `RelevanceConditioned*Estimator`, Point-in-Time Buss-Vilkov & Cornish-Fisher `ImpliedCovarianceEstimator`, Graph Laplacian & Wiener-Hopf Hawkes `ReactionDiffusionCovarianceEstimator`, `AlphaDecayReturnEstimator`, `MultiPeriodMPCAllocator` (CVXPY & ADMM), and `StochasticScenarioMPCAllocator`.
  - **Hierarchical & Graph Machine Learning (`pyalloq.ml`)**: `HRPAllocator`, `SpectralLaplacianHRPAllocator`, `HERCAllocator`, `NCOAllocator`, and Two-Level Bayesian `BayesianHRPAllocator` (RL-BHRP).
  - **Deep Learning & Zero-Shot Foundation Models (`pyalloq.dl`)**: Size-equivariant `SetTransformerBackbone`, `SetTransformerCovarianceNet` (with `O(N m^2 + m^3)` Woodbury losses), `SetTransformerPolicyNet` (5 institutional mandates), `MarketJEPA` self-supervised world model, `ApolloPFNReturnEstimator`, `TimesFM3*`, `Timestral*`, `EnsembleReturnEstimator` / `EnsembleViewGenerator`, `NeuralSDEEstimator`, `UnifiedE2EPolicyNetwork`, `CRISPSparseSTGAT`, `NeuralJumpRDE`, `LogSigSSMBlock`, `DifferentiableRiskParityLayer`, `DifferentiableCVaRLayer`, `MDFPOptimizationLayer` (`O(1)`-memory implicit differentiation), `SPOPlusLoss`, `HybridDFLLoss`, and `IPMOAllocator`.
  - **Reinforcement Learning & Stochastic Control (`pyalloq.rl`)**: `POMDPPortfolioEnv`, `MetaPortfolioEnv`, `PortfolioGymEnv`, `ScenarioPortfolioEnv`, `EndogenousLiquidityImpactWrapper`, `DifferentialSharpeReward` / `StepwiseSortinoReward` / `CARAUtilReward`, `SB3PortfolioAgent` (MLP & Masked-Dirichlet SetTransformer PPO), `HierarchicalMetaRLAllocator`, `WonhamRegimeFilter` & `ExploratoryRegimeMVAllocator`, `DeepHedgingTrajectoryTrainer`, `WassersteinRegimeGate`, `SignatureMMDRegimeGate`, and `RLAllocator`.

## Documentation

Full documentation — including **System Architecture**, **Component Composition Recipes**, **User Guides**, **Verified Mathematics & Notation Glossary**, and **API Reference** — is built with MkDocs Material:

```bash
uv run mkdocs serve
```

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
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
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
