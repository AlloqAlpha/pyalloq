import numpy as np
import pandas as pd
import pytest

from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.engines import ParallelMonteCarloEngine
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_features.core.pipeline import FeaturePipeline
from pyalloq_simulation import (
    AbsoluteView,
    CovarianceView,
    EntropyPooler,
    EvaluationMetrics,
    FactorBackcastImputer,
    FHSGenerator,
    GARCHEVTCopulaGenerator,
    GaussianGenerator,
    GMMRegimeGenerator,
    PCAFactorGenerator,
    ScenarioEvaluator,
    ViewBuilder,
    VolatilityView,
)


class _DummyMomentumFeature(BaseFeatureTransformer):
    def transform(self, data: MarketData) -> MarketData:
        data.features["mom_5"] = data.prices.pct_change(5).fillna(0.0)
        return data


@pytest.mark.parametrize(
    "gen_cls,kwargs",
    [
        (FHSGenerator, {"dist": "Normal"}),
        (PCAFactorGenerator, {"n_components": 2}),
        (GaussianGenerator, {}),
        (GARCHEVTCopulaGenerator, {"nu": 5.0}),
        (GMMRegimeGenerator, {"n_regimes": 2}),
    ],
)
def test_generators_dtype_batching_and_features(
    synthetic_market_data: MarketData, gen_cls, kwargs
) -> None:
    gen = gen_cls(dtype=np.float32, **kwargs)
    gen.fit(synthetic_market_data, log_returns=False)

    scen = gen.generate(n_paths=12, horizon=20)
    assert scen.prices.shape == (12, 20, 4)
    assert scen.returns.shape == (12, 20, 4)
    assert scen.prices.dtype == np.float32
    assert scen.returns.dtype == np.float32
    assert isinstance(scen.timestamps, pd.DatetimeIndex)
    assert scen.initial_prices is not None
    assert "volume" in scen.features
    assert "rsi" in scen.features
    assert scen.features["volume"].shape == (12, 20, 4)

    # Verify batch generation
    batches = list(gen.generate_batches(n_paths=25, horizon=10, batch_size=10))
    assert [b.n_paths for b in batches] == [10, 10, 5]
    for b in batches:
        assert b.prices.dtype == np.float32
        assert b.horizon == 10


def test_factor_backcast_imputer_preserves_full_history(
    synthetic_prices: pd.DataFrame,
) -> None:
    corrupted = synthetic_prices.copy()
    # Simulate an IPO asset missing the first 150 days of history
    corrupted.iloc[:150, 0] = np.nan
    # Simulate interior missing data gaps on another asset
    corrupted.iloc[50:65, 1] = np.nan

    md_missing = MarketData(prices=corrupted)
    # Without imputation, get_returns() drops 150 rows (251 -> 101)
    assert len(md_missing.get_returns()) == 101

    imputer = FactorBackcastImputer(n_components=2)
    md_imputed = imputer.fit_transform(md_missing)

    assert not md_imputed.prices.isna().any().any()
    assert len(md_imputed.get_returns()) == len(synthetic_prices) - 1
    assert (md_imputed.prices > 0).all().all()

    # Generators with impute_missing=True should retain full history automatically
    fhs = FHSGenerator(impute_missing=True)
    fhs.fit(md_missing)
    assert len(fhs.std_residuals) == len(synthetic_prices) - 1


def test_gaussian_generator_handles_collinear_assets(
    collinear_market_data: MarketData,
) -> None:
    gen = GaussianGenerator(dtype=np.float64)
    gen.fit(collinear_market_data, log_returns=True)
    scen = gen.generate(n_paths=50, horizon=15)
    assert scen.prices.shape == (50, 15, 3)
    assert scen.prices.dtype == np.float64
    assert np.all(np.isfinite(scen.prices))


def test_scenario_evaluator_path_dependent_mdd_and_ruin() -> None:
    # Construct 2 deterministic paths over T=3 steps for 1 asset:
    # Path 0: crashes -40% at t=1, then recovers to +10% at t=3 (breaches -30% ruin intra-horizon!)
    # Path 1: steady +5% each step (no drawdown)
    init_p = np.array([100.0])
    prices = np.array(
        [
            [[60.0], [80.0], [110.0]],  # Path 0: MDD = -40%
            [[105.0], [110.0], [115.0]],  # Path 1: MDD = 0%
        ],
        dtype=np.float64,
    )
    returns = np.zeros_like(prices)
    scen = ScenarioMarketData(
        prices=prices,
        returns=returns,
        asset_names=["SPY"],
        initial_prices=init_p,
    )

    pooler = EntropyPooler(n_scenarios=2)
    entropy_res = pooler.fit()  # Uniform weights [0.5, 0.5]

    evaluator = ScenarioEvaluator(ruin_threshold=-0.30)
    metrics: EvaluationMetrics = evaluator.evaluate(
        scen, asset_weights=np.array([1.0]), entropy_result=entropy_res
    )

    # Terminal returns are +10% and +15% -> expected return = +12.5%
    assert np.isclose(metrics.expected_return, 0.125)
    # Path 0 MDD is -0.40, Path 1 MDD is 0.0 -> View-adjusted MDD = -0.20
    assert np.isclose(metrics.view_adjusted_mdd, -0.20)
    # Path 0 breached -30% intra-horizon even though terminal return is +10%!
    assert np.isclose(metrics.probability_of_ruin, 0.50)


def test_volatility_and_covariance_views(
    synthetic_market_data: MarketData,
) -> None:
    np.random.seed(123)
    gen = GaussianGenerator(dtype=np.float64)
    gen.fit(synthetic_market_data)
    scen = gen.generate(n_paths=2000, horizon=20)

    builder = ViewBuilder(scen)
    aapl_rets = builder._get_asset_vector("AAPL")
    msft_rets = builder._get_asset_vector("MSFT")
    prior_aapl_vol = float(np.std(aapl_rets))

    views = [
        AbsoluteView(asset="AAPL", operator="==", target=0.01),
        VolatilityView(asset="AAPL", operator="==", multiplier=1.25),
        CovarianceView(
            asset_1="AAPL", asset_2="MSFT", operator=">=", target_correlation=0.35
        ),
    ]

    A_ineq, b_ineq, C_eq, d_eq = builder.build(views)
    pooler = EntropyPooler(n_scenarios=scen.n_paths)
    res = pooler.fit(A_ineq=A_ineq, b_ineq=b_ineq, C_eq=C_eq, d_eq=d_eq)

    assert res.success
    post_w = res.weights
    post_mean_aapl = float(np.dot(post_w, aapl_rets))
    post_var_aapl = float(np.dot(post_w, (aapl_rets - 0.01) ** 2))
    post_vol_aapl = np.sqrt(post_var_aapl)

    assert np.isclose(post_mean_aapl, 0.01, atol=1e-4)
    assert np.isclose(post_vol_aapl, prior_aapl_vol * 1.25, rtol=1e-2)

    m_msft = float(np.mean(msft_rets))
    post_cov = float(np.dot(post_w, (aapl_rets - 0.01) * (msft_rets - m_msft)))
    expected_min_cov = 0.35 * (prior_aapl_vol * 1.25) * float(np.std(msft_rets))
    assert post_cov >= expected_min_cov - 1e-5


def test_parallel_monte_carlo_engine_integration(
    synthetic_market_data: MarketData,
) -> None:
    gen = GaussianGenerator(dtype=np.float32)
    gen.fit(synthetic_market_data)

    allocator = EqualWeightAllocator(tickers=synthetic_market_data.assets)
    pipeline = StrategyPipeline(allocator=allocator)
    wf_engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=RollingWindowSplitter(lookback_window=10),
        rebalance_freq="ME",
    )

    feat_pipe = FeaturePipeline().add(_DummyMomentumFeature())
    mc_engine = ParallelMonteCarloEngine(
        backtest_engine=wf_engine, max_workers=2, feature_pipeline=feat_pipe
    )

    batches = gen.generate_batches(n_paths=4, horizon=45, batch_size=2)
    mc_result = mc_engine.run_batches(batches)

    assert mc_result.returns_matrix.shape[1] == 4
    assert mc_result.equity_curves.shape[1] == 4
    assert len(mc_result.metrics_distribution) == 4
    assert "sharpe_ratio" in mc_result.metrics_distribution.columns
