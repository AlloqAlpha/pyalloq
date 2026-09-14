import numpy as np
import pandas as pd
import pytest
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline

from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.covariance.random_matrix_theory import (
    RandomMatrixTheoryEstimator,
)
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator

pytestmark = pytest.mark.integration


class TestEndToEndBacktests:
    def test_end_to_end_classical_workflow(
        self, static_market_data: MarketData
    ) -> None:
        """Full end-to-end quant pipeline: static data -> Ledoit-Wolf -> Markowitz -> WalkForwardEngine."""
        assets = list(static_market_data.prices.columns)

        pipeline = StrategyPipeline(
            allocator=MarkowitzAllocator(
                tickers=assets, objective=ObjectiveFunction.MIN_VOLATILITY
            ),
            cov_estimator=LedoitWolfShrinkage(annualization_factor=252.0),
            returns_estimator=EWMAReturnEstimator(span=60),
        )

        engine = WalkForwardEngine(
            pipeline=pipeline,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=FlatBpsCostModel(bps=10.0),
            rebalance_freq="ME",
        )

        results = engine.run(static_market_data)

        assert results is not None
        assert "tear_sheet" in results
        tear_sheet = results["tear_sheet"]
        assert isinstance(tear_sheet, pd.DataFrame)
        assert not np.isnan(tear_sheet.loc["annualized_return", "Value"])
        assert not np.isnan(tear_sheet.loc["sharpe_ratio", "Value"])

        weights = results["weights"].dropna()
        assert not weights.empty
        # Check all non-empty weights sum to 1.0 within tolerance
        for _, row in weights.iterrows():
            assert np.isclose(row.sum(), 1.0, atol=1e-4)

    def test_end_to_end_ml_hrp_workflow(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        """Full end-to-end ML quant pipeline: synthetic data -> RMT Denoising -> HRP Allocator -> Backtest."""
        pipeline = StrategyPipeline(
            allocator=HRPAllocator(tickers=sample_assets),
            cov_estimator=RandomMatrixTheoryEstimator(denoise=True),
        )

        engine = WalkForwardEngine(
            pipeline=pipeline,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=FlatBpsCostModel(bps=5.0),
            rebalance_freq="ME",
        )

        results = engine.run(synthetic_market_data)

        assert results is not None
        assert "tear_sheet" in results
        tear_sheet = results["tear_sheet"]
        assert tear_sheet.loc["maximum_drawdown", "Value"] <= 0.0

        weights = results["weights"].dropna()
        assert not weights.empty
        for _, row in weights.iterrows():
            assert np.isclose(row.sum(), 1.0, atol=1e-4)
