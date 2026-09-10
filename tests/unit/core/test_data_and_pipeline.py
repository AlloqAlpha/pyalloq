import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import OptimizationResult
from pyalloq_core.utils import cov_to_corr

from pyalloq.optimizers.classical.naive import EqualWeightAllocator


class TestMarketData:
    def test_market_data_initialization(
        self, synthetic_prices: pd.DataFrame, sample_assets: list[str]
    ) -> None:
        md = MarketData(prices=synthetic_prices)
        assert md.assets == sample_assets
        assert md.prices.shape == synthetic_prices.shape
        assert md.risk_free_rate == 0.0

    def test_market_data_alignment_validation_feature_mismatch(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        # Create a feature dataframe with mismatched index
        mismatched_index = pd.date_range(
            "2020-01-01", periods=len(synthetic_prices), freq="B"
        )
        bad_feature = pd.DataFrame(
            np.ones((len(synthetic_prices), len(synthetic_prices.columns))),
            index=mismatched_index,
            columns=synthetic_prices.columns,
        )
        with pytest.raises(ValueError, match="Data misalignment: Feature: bad_feat"):
            MarketData(prices=synthetic_prices, features={"bad_feat": bad_feature})

    def test_market_data_alignment_validation_rf_mismatch(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        mismatched_rf = pd.Series(
            0.03, index=pd.date_range("2020-01-01", periods=10, freq="B")
        )
        with pytest.raises(ValueError, match="Data misalignment: 'risk_free_rate'"):
            MarketData(prices=synthetic_prices, risk_free_rate=mismatched_rf)

    def test_market_data_slice_time(self, synthetic_market_data: MarketData) -> None:
        all_dates = synthetic_market_data.prices.index
        midpoint = all_dates[100]
        lookback = 30

        sliced = synthetic_market_data.slice_time(end_date=midpoint, lookback=lookback)
        assert len(sliced.prices) == lookback
        assert sliced.prices.index[-1] == midpoint
        assert len(sliced.features["volume"]) == lookback
        assert sliced.features["volume"].index[-1] == midpoint


class TestOptimizationResult:
    def test_clean_weights(self) -> None:
        weights = pd.Series(
            {"AAPL": 0.49995, "MSFT": 0.49995, "GOOGL": 0.00008, "AMZN": 0.00002}
        )
        res = OptimizationResult(weights=weights, status="OPTIMAL")
        cleaned = res.clean_weights(cutoff=1e-4)

        assert cleaned["GOOGL"] == 0.0
        assert cleaned["AMZN"] == 0.0
        assert np.isclose(cleaned.sum(), 1.0)
        assert np.isclose(cleaned["AAPL"], 0.5)
        assert np.isclose(cleaned["MSFT"], 0.5)


class TestUtils:
    def test_cov_to_corr(self) -> None:
        cov = np.array(
            [
                [4.0, 1.2, -0.8],
                [1.2, 9.0, 0.0],
                [-0.8, 0.0, 1.0],
            ]
        )
        corr = cov_to_corr(cov)

        # Diagonals must be 1.0
        np.testing.assert_allclose(np.diag(corr), np.ones(3), rtol=1e-6)
        # Symmetry
        np.testing.assert_allclose(corr, corr.T, rtol=1e-6)
        # Bounds [-1, 1]
        assert np.all(corr >= -1.0 - 1e-6)
        assert np.all(corr <= 1.0 + 1e-6)
        # Check specific value: corr[0, 1] = 1.2 / (2 * 3) = 0.2
        assert np.isclose(corr[0, 1], 0.2)


class TestStrategyPipeline:
    def test_pipeline_execution(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        allocator = EqualWeightAllocator(tickers=sample_assets)
        pipeline = StrategyPipeline(allocator=allocator)

        weights = pipeline.generate_weights(synthetic_market_data)
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_assets)
        assert np.isclose(weights.sum(), 1.0)
        for asset in sample_assets:
            assert np.isclose(weights[asset], 1.0 / len(sample_assets))
