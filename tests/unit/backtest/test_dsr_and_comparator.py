from __future__ import annotations

import numpy as np
import pandas as pd
from pyalloq_backtest import (
    BacktestTracker,
    BenchmarkSuite,
    ComparisonResult,
    FlatBpsCostModel,
    MultiStrategyComparator,
    RollingWindowSplitter,
    WalkForwardEngine,
    compute_deflated_sharpe_ratio,
    compute_expected_max_sharpe,
)
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_simulation import GMMRegimeGenerator

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator


class TestDeflatedSharpeAndTracker:
    def test_expected_max_sharpe_monotonicity_in_trials(self) -> None:
        sr0_1 = compute_expected_max_sharpe(n_trials=1, sharpe_std=0.5)
        sr0_5 = compute_expected_max_sharpe(n_trials=5, sharpe_std=0.5)
        sr0_50 = compute_expected_max_sharpe(n_trials=50, sharpe_std=0.5)
        sr0_500 = compute_expected_max_sharpe(n_trials=500, sharpe_std=0.5)

        assert sr0_1 == 0.0
        assert 0.0 < sr0_5 < sr0_50 < sr0_500

    def test_dsr_penalizes_multiple_testing_and_non_normality(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, size=252))
        trial_sharpes = [0.2, 0.5, 0.8, 1.1, 1.4]

        dsr_single, sr0_single = compute_deflated_sharpe_ratio(returns, n_trials=1)
        dsr_multi, sr0_multi = compute_deflated_sharpe_ratio(
            returns, n_trials=25, trial_sharpes=trial_sharpes
        )

        assert 0.0 <= dsr_multi <= 1.0
        assert 0.0 <= dsr_single <= 1.0
        assert sr0_multi > sr0_single
        assert dsr_multi < dsr_single

    def test_backtest_tracker_singleton_and_session_scoping(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        t1 = BacktestTracker.instance()
        t2 = BacktestTracker()
        assert t1 is t2

        with BacktestTracker.session(reset=True, restore_on_exit=True) as tracker:
            assert tracker.n_trials == 0

            for span in (10, 20, 40):
                pipe = StrategyPipeline(
                    allocator=MarkowitzAllocator(tickers=sample_assets),
                    returns_estimator=EWMAReturnEstimator(span=span),
                    cov_estimator=EmpiricalCovariance(),
                )
                engine = WalkForwardEngine(
                    pipeline=pipe,
                    splitter=RollingWindowSplitter(lookback_window=60),
                    cost_model=FlatBpsCostModel(bps=5.0),
                    rebalance_freq="ME",
                )
                res = engine.run(synthetic_market_data)
                assert "deflated_sharpe_ratio" in res.tear_sheet.index
                assert "expected_max_sharpe_ratio" in res.tear_sheet.index

            assert tracker.n_trials == 3
            ledger_df = tracker.to_dataframe()
            assert len(ledger_df) == 3
            assert tracker.sharpe_variance(annualized=True) >= 0.0
            assert tracker.expected_max_sharpe() > 0.0


class TestMultiStrategyComparator:
    def test_concurrent_comparison_and_scenario_evaluation(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        assert BenchmarkSuite is MultiStrategyComparator

        pipelines = {
            "EqualWeight": StrategyPipeline(
                allocator=EqualWeightAllocator(tickers=sample_assets)
            ),
            "RiskParity": StrategyPipeline(
                allocator=RiskParityAllocator(tickers=sample_assets),
                cov_estimator=LedoitWolfShrinkage(),
            ),
            "HRP": StrategyPipeline(
                allocator=HRPAllocator(tickers=sample_assets),
                cov_estimator=EmpiricalCovariance(),
            ),
            "Markowitz": StrategyPipeline(
                allocator=MarkowitzAllocator(tickers=sample_assets),
                returns_estimator=EWMAReturnEstimator(span=20),
                cov_estimator=LedoitWolfShrinkage(),
            ),
        }

        gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
        gen.fit(synthetic_market_data)
        scenarios: ScenarioMarketData = gen.generate(n_paths=25, horizon=20)

        with BacktestTracker.session(reset=True, restore_on_exit=True):
            suite = BenchmarkSuite(
                pipelines=pipelines,
                splitter=RollingWindowSplitter(lookback_window=60),
                cost_model=FlatBpsCostModel(bps=10.0),
                rebalance_freq="ME",
                benchmark="EqualWeight",
                max_workers=2,
            )
            comp_res = suite.run(synthetic_market_data, scenario_data=scenarios)

            assert isinstance(comp_res, ComparisonResult)
            assert list(comp_res.summary_table.index) == list(pipelines.keys())
            assert comp_res.n_trials == 4
            assert "deflated_sharpe_ratio" in comp_res.summary_table.columns
            assert "expected_max_sharpe_ratio" in comp_res.summary_table.columns
            assert "brinson_allocation" in comp_res.summary_table.columns
            assert "factor_alpha" in comp_res.summary_table.columns
            assert "total_cost_drag" in comp_res.summary_table.columns

            assert comp_res.scenario_results is not None
            assert set(comp_res.scenario_results.keys()) == set(pipelines.keys())
            assert set(comp_res.attributions.keys()) == set(pipelines.keys())
