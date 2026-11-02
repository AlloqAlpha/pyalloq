import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult, OptimizationResult
from pyalloq_core.utils import cov_to_corr

from pyalloq.classical.optimizers.naive import EqualWeightAllocator


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

    def test_market_data_preserves_risk_aversion_on_slice(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        md_scalar = MarketData(prices=synthetic_prices, risk_aversion=3.5)
        sliced_scalar = md_scalar.slice_time(
            end_date=synthetic_prices.index[50], lookback=20
        )
        assert sliced_scalar.risk_aversion == 3.5

        ra_series = pd.Series(
            np.linspace(1.0, 5.0, len(synthetic_prices)), index=synthetic_prices.index
        )
        md_series = MarketData(prices=synthetic_prices, risk_aversion=ra_series)
        sliced_series = md_series.slice_time(
            end_date=synthetic_prices.index[50], lookback=20
        )
        assert isinstance(sliced_series.risk_aversion, pd.Series)
        assert len(sliced_series.risk_aversion) == 20
        assert sliced_series.risk_aversion.index.equals(sliced_series.prices.index)

        bad_ra = pd.Series(1.0, index=pd.date_range("2019-01-01", periods=5, freq="B"))
        with pytest.raises(ValueError, match="Data misalignment: 'risk_aversion'"):
            MarketData(prices=synthetic_prices, risk_aversion=bad_ra)


class TestOptimizationResult:
    def test_clean_weights(self) -> None:
        weights = pd.Series(
            {"AAPL": 0.49995, "MSFT": 0.49995, "GOOGL": 0.00008, "AMZN": 0.00002}
        )
        res = OptimizationResult(weights=weights, status="OPTIMAL", name="TestOpt")
        assert res.name == "TestOpt"
        cleaned = res.clean_weights(cutoff=1e-4)

        assert cleaned["GOOGL"] == 0.0
        assert cleaned["AMZN"] == 0.0
        assert np.isclose(cleaned.sum(), 1.0)
        assert np.isclose(cleaned["AAPL"], 0.5)
        assert np.isclose(cleaned["MSFT"], 0.5)

    def test_clean_weights_zero_sum_safe(self) -> None:
        zero_weights = pd.Series({"AAPL": 0.0, "MSFT": 0.0})
        res = OptimizationResult(weights=zero_weights, status="INFEASIBLE")
        assert res.name == ""
        cleaned = res.clean_weights()
        assert not cleaned.isna().any()
        assert (cleaned == 0.0).all()

    def test_resolve_annualization_factor_string_and_enum(
        self, synthetic_prices: pd.DataFrame, sample_assets: list[str]
    ) -> None:
        allocator = EqualWeightAllocator(tickers=sample_assets)
        returns = synthetic_prices.pct_change().dropna()
        assert allocator._resolve_annualization_factor(returns, "WEEKLY") == 52.0
        assert allocator._resolve_annualization_factor(returns, "monthly") == 12.0
        assert allocator._resolve_annualization_factor(returns, "CRYPTO_DAILY") == 365.0
        assert allocator._resolve_annualization_factor(returns, 100) == 100.0


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


class TestBacktestResult:
    def test_backtest_result_mapping_and_attribute_access(self) -> None:
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015], index=dates)
        weights = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=dates)
        tear_sheet = pd.DataFrame(
            {"Value": [0.15, 1.2]}, index=["annualized_return", "sharpe_ratio"]
        )

        res = BacktestResult(
            name="TestStrategy",
            returns=returns,
            weights=weights,
            tear_sheet=tear_sheet,
        )

        # Attribute access
        assert res.name == "TestStrategy"
        assert res.returns is returns
        assert res.weights is weights
        assert res.tear_sheet is tear_sheet
        assert isinstance(res.equity_curve, pd.Series)

        # Dict / Mapping access
        assert "returns" in res
        assert "weights" in res
        assert "tear_sheet" in res
        assert "equity_curve" in res
        assert "name" in res
        assert "nonexistent" not in res

        assert res["name"] == "TestStrategy"
        assert res["returns"] is returns
        assert res["weights"] is weights
        assert res["tear_sheet"] is tear_sheet
        assert isinstance(res["equity_curve"], pd.Series)

        # get method
        assert res.get("name") == "TestStrategy"
        assert res.get("missing", "default") == "default"

        # keys, items, values, len, iter
        assert set(res.keys()) == {
            "name",
            "returns",
            "weights",
            "tear_sheet",
            "equity_curve",
        }
        assert len(res) == 5
        assert set(iter(res)) == {
            "name",
            "returns",
            "weights",
            "tear_sheet",
            "equity_curve",
        }

        with pytest.raises(KeyError):
            _ = res["missing"]
