import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.estimators.returns.classical.implied import (
    ImpliedReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.james_stein import (
    JamesSteinReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.mean import (
    MeanHistoricalReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.momentum import (
    ResidualMomentumEstimator,
    VolatilityScaledMultiHorizonEstimator,
)


class TestReturnEstimators:
    def test_ewma_returns(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        estimator = EWMAReturnEstimator(span=60)
        returns = estimator.estimate(synthetic_market_data)

        assert isinstance(returns, pd.Series)
        assert list(returns.index) == sample_assets
        assert not returns.isna().any()

    def test_james_stein_shrinkage_property(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        estimator = JamesSteinReturnEstimator()
        js_returns = estimator.estimate(synthetic_market_data)

        # Historical sample mean returns
        raw_rets = synthetic_market_data.prices.pct_change().dropna().mean() * 252

        # James-Stein shrinks towards the grand mean: variance of JS estimates <= variance of raw sample means
        assert np.var(js_returns.values) <= np.var(raw_rets.values) + 1e-6
        assert list(js_returns.index) == sample_assets

    def test_volatility_scaled_momentum_requires_sufficient_history(
        self, sample_assets: list[str]
    ) -> None:
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        df = pd.DataFrame(100.0, index=dates, columns=sample_assets)
        short_data = MarketData(prices=df)

        estimator = VolatilityScaledMultiHorizonEstimator(horizons=(63, 126, 252))
        with pytest.raises(
            ValueError, match="Requires at least 252 price observations"
        ):
            estimator.estimate(short_data)

    def test_volatility_scaled_momentum_properties(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        estimator = VolatilityScaledMultiHorizonEstimator(
            horizons=(20, 40, 60), skip_period=5
        )
        scores = estimator.estimate(synthetic_market_data)

        assert isinstance(scores, pd.Series)
        assert list(scores.index) == sample_assets
        assert not scores.isna().any()
        # Cross-sectional standardized average z-scores mean ~ 0
        np.testing.assert_allclose(scores.mean(), 0.0, atol=1e-5)

    def test_residual_momentum_requires_sufficient_history(
        self, sample_assets: list[str]
    ) -> None:
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        df = pd.DataFrame(100.0, index=dates, columns=sample_assets)
        short_data = MarketData(prices=df)

        estimator = ResidualMomentumEstimator(lookback=252)
        with pytest.raises(
            ValueError, match="Requires at least 252 price observations"
        ):
            estimator.estimate(short_data)

    def test_residual_momentum_properties(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        estimator = ResidualMomentumEstimator(lookback=120, skip_period=10)
        scores = estimator.estimate(synthetic_market_data)

        assert isinstance(scores, pd.Series)
        assert list(scores.index) == sample_assets
        assert not scores.isna().any()
        assert scores.name == "expected_returns"
        np.testing.assert_allclose(scores.mean(), 0.0, atol=1e-5)

    def test_implied_returns(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov_est = EmpiricalCovariance()
        market_caps = pd.DataFrame(
            {"market_cap": [1e9, 2e9, 1.5e9, 0.5e9]},
            index=sample_assets,
        )
        synthetic_market_data.cross_sectional = market_caps

        implied_est = ImpliedReturnEstimator(cov_estimator=cov_est, risk_aversion=2.5)
        pi = implied_est.estimate(synthetic_market_data)

        assert isinstance(pi, pd.Series)
        assert list(pi.index) == sample_assets
        assert not pi.isna().any()

    def test_capm_returns(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        from pyalloq.classical.estimators.returns.classical.factor import (
            CAPMReturnEstimator,
        )

        # Market prices feature
        market_prices = pd.DataFrame(
            {"SPY": synthetic_market_data.prices.mean(axis=1)},
            index=synthetic_market_data.prices.index,
        )
        synthetic_market_data.features["market_prices"] = market_prices

        capm = CAPMReturnEstimator(risk_free_rate=0.03, market_risk_premium=0.06)
        mu = capm.estimate(synthetic_market_data)

        assert isinstance(mu, pd.Series)
        assert list(mu.index) == sample_assets
        assert not mu.isna().any()

    def test_multifactor_returns(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        from pyalloq.classical.estimators.returns.classical.factor import (
            MultiFactorReturnEstimator,
        )

        factor_data = pd.DataFrame(
            np.random.normal(0.0005, 0.01, size=(len(synthetic_market_data.prices), 2)),
            index=synthetic_market_data.prices.index,
            columns=["Factor1", "Factor2"],
        )
        synthetic_market_data.cross_sectional = factor_data
        factor_premium = pd.Series([0.04, 0.02], index=["Factor1", "Factor2"])

        mf = MultiFactorReturnEstimator(
            factor_premium=factor_premium, risk_free_rate=0.03
        )
        mu = mf.estimate(synthetic_market_data)

        assert isinstance(mu, pd.Series)
        assert list(mu.index) == sample_assets
        assert not mu.isna().any()

    def test_mean_historical_return_estimator_arithmetic_and_cagr(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        arith_est = MeanHistoricalReturnEstimator(compounding=False, frequency="DAILY")
        cagr_est = MeanHistoricalReturnEstimator(compounding=True, frequency="DAILY")
        weekly_est = MeanHistoricalReturnEstimator(
            compounding=False, frequency="WEEKLY"
        )

        mu_arith = arith_est.estimate(synthetic_market_data)
        mu_cagr = cagr_est.estimate(synthetic_market_data)
        mu_weekly = weekly_est.estimate(synthetic_market_data)

        assert list(mu_arith.index) == sample_assets
        assert list(mu_cagr.index) == sample_assets
        # AM-GM inequality: geometric mean <= compounded arithmetic mean
        compounded_arith = (1.0 + mu_arith.values / 252.0) ** 252.0 - 1.0
        assert np.all(mu_cagr.values <= compounded_arith + 1e-6)
        # Weekly scale factor (52) vs Daily (252)
        np.testing.assert_allclose(
            mu_weekly.values, mu_arith.values * (52.0 / 252.0), rtol=1e-5
        )
