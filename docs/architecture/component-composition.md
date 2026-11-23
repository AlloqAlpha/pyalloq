# Component Composition Recipes

Because every component in PyAlloq adheres to a narrow `Base*` contract, you can assemble sophisticated, multi-paradigm strategies by snapping blocks together. Below are **seven complete, end-to-end composition patterns** spanning classical, regime-aware, meta-ensemble, foundation-model, deep-learning, reinforcement-learning, and multiverse stress-testing stacks.

---

## Recipe 1: Classical Pipeline with Jump-Model Black-Litterman & Discrete Execution

Combine `FeaturePipeline`, `AssetSpecificJumpViewGenerator` (per-asset Sparse Jump Model + boosted regime forecasting), `BlackLittermanReturnEstimator`, `LedoitWolfShrinkage`, `MarkowitzAllocator`, and `DiscreteShareAllocator` (MILP integer share rounding).

```python
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq_features import FeaturePipeline, TechnicalIndicator, RollingZScoreScaler
from pyalloq.classical.estimators.black_litterman.jump_views import AssetSpecificJumpViewGenerator
from pyalloq.classical.estimators.black_litterman.bayesian import BlackLittermanReturnEstimator
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.discrete import DiscreteShareAllocator
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_backtest.costs import AlmgrenChrissCostModel

# 1. Ingest & engineer point-in-time features
raw_data = YahooFinanceClient().get_market_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM"],
    start="2019-01-01",
    end="2025-01-01",
)
feat_pipe = (
    FeaturePipeline()
    .add(TechnicalIndicator(indicator="rsi", timeperiod=14))
    .add(RollingZScoreScaler(window=60))
)
data = feat_pipe.run(raw_data)

# 2. Wire Jump-Model Black-Litterman views into Markowitz
jump_views = AssetSpecificJumpViewGenerator(n_regimes=2, jump_penalty=15.0)
bl_returns = BlackLittermanReturnEstimator(view_generator=jump_views, tau=0.05)

pipeline = StrategyPipeline(
    returns_estimator=bl_returns,
    cov_estimator=LedoitWolfShrinkage(),
    allocator=MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_SHARPE,
        weight_bounds=(0.0, 0.35),
        max_turnover=0.25,
    ),
)

# 3. Backtest with Almgren-Chriss market impact
engine = WalkForwardEngine(
    pipeline=pipeline,
    rebalance_freq="ME",
    splitter=RollingWindowSplitter(lookback_window=252),
    cost_model=AlmgrenChrissCostModel(portfolio_aum=5_000_000.0, spread_bps=2.0, gamma=0.1),
)
result = engine.run(data)

# 4. Convert latest continuous target weights into integer share orders
latest_weights = result.target_weights.iloc[-1]
latest_prices = data.prices.iloc[-1]
orders = DiscreteShareAllocator(total_portfolio_value=500_000.0, lot_size=1).allocate(
    weights=latest_weights,
    latest_prices=latest_prices,
    method="lp",
)
print(orders.shares, f"Leftover cash: ${orders.leftover_cash:,.2f}")
```

---

## Recipe 2: Regime-Adaptive Distributionally Robust + Conformal Safety Gate

Layer a **forward-looking covariance estimator** (`ImpliedCovarianceEstimator` or `ReactionDiffusionCovarianceEstimator`), a **regime-adaptive `WassersteinDROAllocator`**, and a **`ConformalSafetyGateAllocator`** that guarantees finite-sample tail-loss, volatility, and turnover certificates via convex Euclidean projection.

```python
from pyalloq_features.transformers.alternative import PITOptionsMFIVTransformer
from pyalloq_features.transformers.rough_signatures import LogSignatureTransformer
from pyalloq_features import FeaturePipeline
from pyalloq.classical.estimators.covariance.implied_covariance import ImpliedCovarianceEstimator
from pyalloq.classical.optimizers.wasserstein_dro import WassersteinDROAllocator
from pyalloq.classical.optimizers.conformal import ConformalSafetyGateAllocator, RiskCertificate
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine

# 1. Attach PIT implied volatility & path log-signatures
data = (
    FeaturePipeline()
    .add(PITOptionsMFIVTransformer(fallback_window=20))
    .add(LogSignatureTransformer(window=10, depth=2))
    .run(raw_data)
)

# 2. Inner robust allocator: Wasserstein-Kelly DRO with quantile Mahalanobis ground metric
dro_allocator = WassersteinDROAllocator(
    tickers=data.assets,
    objective=ObjectiveFunction.WASSERSTEIN_KELLY,
    epsilon=0.005,
    ground_metric="quantile_mahalanobis",
    radius_mode="regime_adaptive",
    kelly_fraction=0.5,
    weight_bounds=(0.0, 0.35),
)

# 3. Outer safety gate: Gated-CSRC finite-sample risk certificate
certificate = RiskCertificate(
    max_tail_loss=0.035,
    max_annual_volatility=0.18,
    max_turnover=0.30,
    max_asset_weight=0.35,
    min_cash_buffer=0.0,
)
safe_allocator = ConformalSafetyGateAllocator(
    base_allocator=dro_allocator,
    certificate=certificate,
    alpha=0.15,
    method="pid",
)

pipeline = StrategyPipeline(
    cov_estimator=ImpliedCovarianceEstimator(correlation_window=60, use_cornish_fisher_tail=True),
    allocator=safe_allocator,
)
result = WalkForwardEngine(pipeline=pipeline, rebalance_freq="ME").run(data)
```

---

## Recipe 3: Multi-Strategy Meta-Ensemble & Unified Comparison Tear Sheet

Combine diverse child allocators (Classical + Graph ML + DRO) inside `EnsembleAllocator` using switching-cost online learning (`DualScaledSwitchingCostStrategy`) with pre-TCA internal trade crossing, then benchmark the ensemble against its constituent children using `MultiStrategyComparator` and `ComparisonTearSheet`.

```python
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.classical.optimizers.wasserstein_dro import WassersteinDROAllocator
from pyalloq.classical.optimizers.ensemble import EnsembleAllocator
from pyalloq.ml.optimizers.hrp import SpectralLaplacianHRPAllocator
from pyalloq_core.meta import DualScaledSwitchingCostStrategy
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.benchmark import MultiStrategyComparator
from pyalloq_viz.tearsheets.comparison_ts import ComparisonTearSheet

children = [
    RiskParityAllocator(tickers=data.assets),
    SpectralLaplacianHRPAllocator(tickers=data.assets),
    WassersteinDROAllocator(tickers=data.assets),
]

ensemble = EnsembleAllocator(
    child_allocators=children,
    meta_strategy=DualScaledSwitchingCostStrategy(switching_cost_penalty=0.25),
    include_1n_anchor=True,
    evaluation_window=60,
    switching_cost_bps=10.0,
    tickers=data.assets,
)

strategies = {
    "RiskParity": StrategyPipeline(allocator=children[0]),
    "SpectralHRP": StrategyPipeline(allocator=children[1]),
    "WassersteinDRO": StrategyPipeline(allocator=children[2]),
    "MetaEnsemble": StrategyPipeline(allocator=ensemble),
}

comparator = MultiStrategyComparator(rebalance_freq="ME")
comparison = comparator.compare(data=data, strategies=strategies)

ts = ComparisonTearSheet(comparison, market_data=data)
ts.to_html("comparison_report.html", theme="alloq_dark")
```

---

## Recipe 4: Zero-Shot Foundation Model Ensemble + Bayesian View Pooling

Blend multiple zero-shot time-series foundation models (`ApolloPFNReturnEstimator`, `TimesFM3MultivariateReturnEstimator`, `TimestralMultivariateReturnEstimator`) with `sCRPS`-adaptive Bayesian weighting (`EnsembleReturnEstimator`), or fuse their Black-Litterman views via exact precision pooling (`EnsembleViewGenerator`).

```python
from pyalloq.dl.estimators.apollopfn import ApolloPFNReturnEstimator, ApolloPFNViewGenerator
from pyalloq.dl.estimators.returns.timesfm import TimesFM3MultivariateReturnEstimator
from pyalloq.dl.estimators.black_litterman.timesfm import TimesFM3MultivariateViewGenerator
from pyalloq.dl.estimators.ensemble import EnsembleReturnEstimator, EnsembleViewGenerator
from pyalloq.classical.estimators.black_litterman.bayesian import BlackLittermanReturnEstimator
from pyalloq.classical.optimizers.cvar import MeanCVaRAllocator
from pyalloq_core.enums import CovariateMode
from pyalloq_core.pipeline import StrategyPipeline

# Option A: Direct sCRPS-adaptive return & scenario ensemble
ens_returns = EnsembleReturnEstimator(
    estimators=[
        ApolloPFNReturnEstimator(horizon=21, covariate_mode=CovariateMode.AUTO),
        TimesFM3MultivariateReturnEstimator(horizon=21, covariate_mode=CovariateMode.AUTO),
    ],
    weighting_mode="scrps_adaptive",
    validation_horizon=10,
)

# Option B: Precision-pooled Black-Litterman views (Omega_ens = (sum Omega_m^-1)^-1)
ens_views = EnsembleViewGenerator(
    view_generators=[
        ApolloPFNViewGenerator(horizon=21, uncertainty_mode="cvar_tail"),
        TimesFM3MultivariateViewGenerator(horizon=21, uncertainty_mode="cvar_tail"),
    ]
)
bl_ens = BlackLittermanReturnEstimator(view_generator=ens_views, tau=0.05)

pipeline = StrategyPipeline(
    returns_estimator=bl_ens,  # or ens_returns
    allocator=MeanCVaRAllocator(tickers=data.assets, beta=0.95),
)
```

---

## Recipe 5: Size-Equivariant Deep Learning Covariance Pipeline (`SetTransformer` + Woodbury Loss)

Train a permutation- and size-equivariant `SetTransformerCovarianceNet` using the \(O(Nm^2 + m^3)\) `WoodburyGaussianNLLLoss` and `UniverseSubsampler` via `WalkForwardTrainer`, then deploy it directly inside `StrategyPipeline` with `DeepCovarianceEstimator`.

```python
import torch
from pyalloq.dl.estimators.covariance.models.set_covariance import SetTransformerCovarianceNet
from pyalloq.dl.estimators.covariance.deep_cov import DeepCovarianceEstimator
from pyalloq.dl.loss_functions.woodbury import WoodburyGaussianNLLLoss
from pyalloq.dl.dataset import UniverseSubsampler
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.pipeline import StrategyPipeline

# 1. Define permutation-equivariant factor+diagonal covariance network
cov_net = SetTransformerCovarianceNet(
    n_features=1,
    hidden_dim=64,
    num_heads=4,
    num_inds=16,
    head_type="factor_diag",
    use_ssmax=True,
)

# 2. Causal walk-forward training using O(N m^2 + m^3) Woodbury NLL
trainer = WalkForwardTrainer(
    model=cov_net,
    loss_function=WoodburyGaussianNLLLoss(),
    optimizer=torch.optim.AdamW(cov_net.parameters(), lr=1e-3),
    splitter=RollingWindowSplitter(lookback_window=252),
    lookback_window=60,
    horizon=1,
    batch_size=32,
    epochs_per_window=5,
    return_mask=True,
    allow_ragged=True,
    universe_subsampler=UniverseSubsampler(min_assets=4),
    use_forward_factors=True,
)
rebal_dates = data.prices.resample("QE").last().index[4:]
history = trainer.train(data, rebalance_dates=rebal_dates)

# 3. Drop into StrategyPipeline with any classical or ML allocator
pipeline = StrategyPipeline(
    cov_estimator=DeepCovarianceEstimator(model=cov_net, lookback_window=60, allow_ragged=True),
    allocator=RiskParityAllocator(tickers=data.assets),
)
```

---

## Recipe 6: Multiverse RL with SetTransformer PPO, Endogenous Impact & Thermodynamic Gate

Generate synthetic crash/regime paths with `GARCHEVTCopulaGenerator`, wrap `POMDPPortfolioEnv` with `EndogenousLiquidityImpactWrapper` and `FeatureNormalizationWrapper`, train a native Masked-Dirichlet `SetTransformerPolicyNet` via `SB3PortfolioAgent`, and deploy it inside `WalkForwardEngine` behind a `WassersteinRegimeGate`.

```python
from pyalloq_simulation.generators.copulas import GARCHEVTCopulaGenerator
from pyalloq.dl.estimators.e2e.models.set_policy import SetTransformerPolicyNet
from pyalloq.rl.envs.pomdp_env import POMDPPortfolioEnv
from pyalloq.rl.rewards.functions import DifferentialSharpeReward
from pyalloq.rl.wrappers.endogenous_impact import EndogenousLiquidityImpactWrapper
from pyalloq.rl.wrappers.dict_features import FeatureNormalizationWrapper
from pyalloq.rl.agents.sb3_wrapper import SB3PortfolioAgent
from pyalloq.rl.allocator import RLAllocator
from pyalloq.rl.thermodynamics import WassersteinRegimeGate
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine

# 1. Fit fat-tailed GARCH-EVT-Student-t copula and generate training multiverse
gen = GARCHEVTCopulaGenerator(nu=5.0)
gen.fit(data)
train_scenarios = gen.generate(n_paths=100, horizon=126)

# 2. Build POMDP environment with endogenous order-book impact
env = POMDPPortfolioEnv(
    scenario_data=train_scenarios,
    lookback_window=20,
    include_cash=True,
    projection_mode="dirichlet",
    cost_model=FlatBpsCostModel(bps=10.0),
    reward_function=DifferentialSharpeReward(drawdown_penalty=0.15),
)
env = EndogenousLiquidityImpactWrapper(env, impact_lambda=0.005, resilience_rho=0.65)
env = FeatureNormalizationWrapper(env)

# 3. Train Masked-Dirichlet SetTransformer PPO agent
policy_net = SetTransformerPolicyNet(
    n_features=1,
    hidden_dim=64,
    mandate="masked_dirichlet",
    include_cash=True,
)
agent = SB3PortfolioAgent(
    env=env,
    algorithm="PPO",
    policy_backbone="set_transformer",
    set_policy_net=policy_net,
)
agent.fit(total_timesteps=4096)

# 4. Deploy causally in WalkForwardEngine with Wasserstein thermodynamic damping
rl_allocator = RLAllocator(
    policy=agent,
    lookback_window=20,
    use_pomdp_obs=True,
    include_cash=True,
    projection_mode="dirichlet",
    regime_gate=WassersteinRegimeGate(w2_threshold=0.03, sensitivity=25.0),
    tickers=data.assets,
)
result = WalkForwardEngine(pipeline=StrategyPipeline(allocator=rl_allocator)).run(data)
```

---

## Recipe 7: Multiverse Stress-Testing, Entropy Pooling & Overfitting Audit (CPCV/PBO)

Combine `GARCHEVTCopulaGenerator`, Meucci `EntropyPooler` + `ViewBuilder`, `ParallelMonteCarloEngine`, `CPCVSplitter` + `PBOEvaluator`, and generate interactive `SimulationTearSheet` and `RobustnessTearSheet` HTML dashboards.

```python
from pyalloq_simulation.generators.copulas import GARCHEVTCopulaGenerator
from pyalloq_simulation.views.builder import ViewBuilder, VolatilityView, RelativeView
from pyalloq_simulation.views.pooler import EntropyPooler
from pyalloq_simulation.core.evaluator import ScenarioEvaluator
from pyalloq_core.engines import ParallelMonteCarloEngine
from pyalloq_backtest.overfitting import CPCVSplitter, PBOEvaluator
from pyalloq_viz.tearsheets.simulation_ts import SimulationTearSheet
from pyalloq_viz.tearsheets.robustness_ts import RobustnessTearSheet

# 1. Generate synthetic multiverse & apply Meucci Entropy Pooling views
gen = GARCHEVTCopulaGenerator(nu=5.0)
gen.fit(data)
scenarios = gen.generate(n_paths=500, horizon=252)

views = (
    ViewBuilder(scenarios)
    .add(VolatilityView(asset="AAPL", target_vol=0.35, operator=">="))
    .add(RelativeView(asset_long="MSFT", asset_short="XOM", target_spread=0.05, operator=">="))
    .build()
)
ep_result = EntropyPooler().solve(views)

# 2. Run WalkForwardEngine across all 500 paths in parallel
mc_engine = ParallelMonteCarloEngine(backtest_engine=engine, max_workers=4)
mc_result = mc_engine.run(scenarios)

eval_metrics = ScenarioEvaluator().evaluate(mc_result, posterior_probs=ep_result.posterior_probs)
SimulationTearSheet(mc_result, entropy_result=ep_result).to_html("simulation_ts.html")

# 3. Audit backtest overfitting via Combinatorial Purged CV (CPCV) & PBO
cpcv = CPCVSplitter(n_groups=6, n_test_groups=2, purge_pct=0.01, embargo_pct=0.01)
pbo_result = PBOEvaluator(splitter=cpcv).evaluate(data=data, pipelines=list(strategies.values()))
RobustnessTearSheet(pbo_result).to_html("robustness_ts.html")
```
