from typing import Any

import numpy as np
import pandas as pd
import pytest
from pyalloq_core import ObjectiveFunction, StrategyPipeline
from pyalloq_core.data import MarketData

import pyalloq.dl.estimators.returns.timestral as timestral_ret_module
from pyalloq.classical import (
    BlackLittermanCovarianceEstimator,
    BlackLittermanEstimator,
    BlackLittermanReturnEstimator,
    LedoitWolfShrinkage,
    MarkowitzAllocator,
)
from pyalloq.dl import (
    TimestralMultivariateReturnEstimator,
    TimestralMultivariateViewGenerator,
    TimestralUnivariateReturnEstimator,
    TimestralUnivariateViewGenerator,
)

QUANTILE_LEVELS = (
    0.01,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
)


class MockTimestralForecast:
    """Mock of Timestral's forecast return object with 21 non-crossing quantiles."""

    def __init__(self, base_trajectory: np.ndarray) -> None:
        # base_trajectory shape: (batch, variates, horizon)
        self._median = base_trajectory.astype(np.float32)
        self.quantile_levels = QUANTILE_LEVELS
        # Build 21 non-crossing quantiles around median
        quantiles = np.zeros(
            (*base_trajectory.shape, len(QUANTILE_LEVELS)), dtype=np.float32
        )
        for idx, q in enumerate(QUANTILE_LEVELS):
            multiplier = 1.0 + (q - 0.50) * 0.125  # q=0.1 -> 0.95, q=0.9 -> 1.05
            quantiles[..., idx] = self._median * multiplier
        self.quantiles = quantiles

    @property
    def median(self) -> np.ndarray:
        return self._median

    def quantile(self, q: float) -> np.ndarray:
        for idx, level in enumerate(self.quantile_levels):
            if np.isclose(q, level):
                return self.quantiles[..., idx]
        # Linear interpolation fallback for arbitrary quantile in (0, 1)
        multiplier = 1.0 + (q - 0.50) * 0.125
        return self._median * multiplier


class MockTimestralForecaster:
    """Mock of TimestralForecaster recording input shapes and covariates."""

    def __init__(self, horizon: int = 21) -> None:
        self.horizon = horizon
        self.last_context: np.ndarray | None = None
        self.last_past_covariates: np.ndarray | None = None
        self.last_known_covariates: np.ndarray | None = None
        self.last_prediction_length: int | None = None

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int = 21,
        past_covariates: np.ndarray | None = None,
        known_covariates: np.ndarray | None = None,
        **kwargs: Any,
    ) -> MockTimestralForecast:
        assert context.ndim == 3, "Timestral expects 3D context (batch, variates, time)"
        self.last_context = context
        self.last_past_covariates = past_covariates
        self.last_known_covariates = known_covariates
        self.last_prediction_length = prediction_length

        batch_size, n_variates, t_steps = context.shape
        if past_covariates is not None:
            assert past_covariates.ndim == 3
            assert past_covariates.shape[0] == batch_size
            assert past_covariates.shape[2] == t_steps

        if known_covariates is not None:
            assert known_covariates.ndim == 3
            assert known_covariates.shape[0] == batch_size
            assert known_covariates.shape[2] == t_steps + prediction_length

        last_vals = context[:, :, -1]  # shape (batch, variates)
        base_traj = np.zeros(
            (batch_size, n_variates, prediction_length), dtype=np.float32
        )
        for b in range(batch_size):
            for v in range(n_variates):
                base_traj[b, v, :] = np.linspace(
                    last_vals[b, v] * 1.01,
                    last_vals[b, v] * 1.05,
                    prediction_length,
                )
        return MockTimestralForecast(base_traj)


@pytest.mark.dl
class TestTimestralEstimators:
    def test_univariate_return_estimator(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockTimestralForecaster(horizon=horizon)
        estimator = TimestralUnivariateReturnEstimator(
            model=model, horizon=horizon, annualize=True
        )

        returns = estimator.estimate(synthetic_market_data)
        n_assets = len(synthetic_market_data.assets)
        t_steps = len(synthetic_market_data.prices)

        assert model.last_context is not None
        assert model.last_context.shape == (n_assets, 1, t_steps)
        assert isinstance(returns, pd.Series)
        assert list(returns.index) == synthetic_market_data.assets
        assert returns.name == "expected_returns"
        assert not returns.isna().any()
        assert np.allclose(returns.values, 0.05 * (252.0 / horizon), rtol=1e-4)

        # Non-annualized check
        raw_estimator = TimestralUnivariateReturnEstimator(
            model=model, horizon=horizon, annualize=False
        )
        raw_returns = raw_estimator.estimate(synthetic_market_data)
        assert np.allclose(raw_returns.values, 0.05, rtol=1e-4)

    def test_multivariate_return_estimator_with_past_and_future_covariates(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        t_steps = len(synthetic_market_data.prices)
        assets = synthetic_market_data.assets
        n_assets = len(assets)

        calendar_cov = pd.DataFrame(
            np.ones((t_steps, n_assets)),
            index=synthetic_market_data.prices.index,
            columns=assets,
        )
        data_with_future = MarketData(
            prices=synthetic_market_data.prices,
            features=synthetic_market_data.features,
            future_features={"calendar": calendar_cov},
            risk_free_rate=synthetic_market_data.risk_free_rate,
        )

        model = MockTimestralForecaster(horizon=horizon)
        estimator = TimestralMultivariateReturnEstimator(
            model=model, horizon=horizon, annualize=True
        )

        returns = estimator.estimate(data_with_future)
        assert model.last_context is not None
        assert model.last_context.shape == (1, n_assets, t_steps)
        assert model.last_past_covariates is not None
        assert model.last_past_covariates.shape == (1, 2 * n_assets, t_steps)
        assert model.last_known_covariates is not None
        assert model.last_known_covariates.shape == (
            1,
            1 * n_assets,
            t_steps + horizon,
        )

        assert isinstance(returns, pd.Series)
        assert list(returns.index) == assets
        assert not returns.isna().any()
        assert (returns > 0).all()

    def test_univariate_view_generator_and_confidence_intervals(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockTimestralForecaster(horizon=horizon)
        generator_default = TimestralUnivariateViewGenerator(
            model=model, horizon=horizon, confidence_interval=(0.10, 0.90)
        )

        P, Q, Omega = generator_default.generate(synthetic_market_data)
        n_assets = len(synthetic_market_data.assets)

        assert isinstance(P, pd.DataFrame)
        assert np.allclose(P.values, np.eye(n_assets))
        assert isinstance(Q, pd.Series)
        assert len(Q) == n_assets
        assert not Q.isna().any()
        assert isinstance(Omega, pd.DataFrame)
        assert Omega.shape == (n_assets, n_assets)
        diag = np.diag(np.diag(Omega.values))
        assert np.allclose(Omega.values, diag)
        assert (np.diag(Omega.values) > 0).all()

        # Test custom confidence interval (0.05, 0.95) using Timestral's 21 quantiles
        generator_wide = TimestralUnivariateViewGenerator(
            model=model, horizon=horizon, confidence_interval=(0.05, 0.95)
        )
        _P_w, Q_w, Omega_w = generator_wide.generate(synthetic_market_data)
        assert np.allclose(Q.values, Q_w.values)
        assert (np.diag(Omega_w.values) > 0).all()

        # Invalid confidence interval raises ValueError
        with pytest.raises(ValueError, match="confidence_interval"):
            TimestralUnivariateViewGenerator(
                model=model, horizon=horizon, confidence_interval=(0.90, 0.10)
            )

    def test_multivariate_view_generator_with_covariates(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        t_steps = len(synthetic_market_data.prices)
        assets = synthetic_market_data.assets
        n_assets = len(assets)

        calendar_cov = pd.DataFrame(
            np.zeros((t_steps, n_assets)),
            index=synthetic_market_data.prices.index,
            columns=assets,
        )
        data_with_future = MarketData(
            prices=synthetic_market_data.prices,
            features=synthetic_market_data.features,
            future_features={"calendar": calendar_cov},
        )

        model = MockTimestralForecaster(horizon=horizon)
        generator = TimestralMultivariateViewGenerator(
            model=model, horizon=horizon, confidence_interval=(0.10, 0.90)
        )

        P, Q, Omega = generator.generate(data_with_future)
        assert model.last_context is not None
        assert model.last_context.shape == (1, n_assets, t_steps)
        assert model.last_past_covariates is not None
        assert model.last_past_covariates.shape == (1, 2 * n_assets, t_steps)
        assert model.last_known_covariates is not None
        assert model.last_known_covariates.shape == (
            1,
            1 * n_assets,
            t_steps + horizon,
        )

        assert isinstance(P, pd.DataFrame)
        assert np.allclose(P.values, np.eye(n_assets))
        assert isinstance(Q, pd.Series)
        assert len(Q) == n_assets
        assert isinstance(Omega, pd.DataFrame)
        assert Omega.shape == (n_assets, n_assets)
        assert (np.diag(Omega.values) > 0).all()

    def test_lazy_loading_and_missing_import_error(
        self, synthetic_market_data: MarketData, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1. Verify informative ImportError when timestral is not installed
        monkeypatch.setattr(timestral_ret_module, "TimestralForecaster", None)
        est_missing = TimestralUnivariateReturnEstimator(model="mistralai/Timestral-5M")
        with pytest.raises(ImportError, match="mistralai/timestral.git"):
            est_missing.estimate(synthetic_market_data)

        # 2. Verify lazy-loading via from_pretrained when TimestralForecaster is present
        loaded_calls: list[tuple[str, dict[str, Any]]] = []

        class DummyForecasterClass:
            @classmethod
            def from_pretrained(cls, checkpoint: str, **kwargs: Any) -> Any:
                loaded_calls.append((checkpoint, kwargs))
                return MockTimestralForecaster(horizon=21)

        monkeypatch.setattr(
            timestral_ret_module, "TimestralForecaster", DummyForecasterClass
        )
        est_lazy = TimestralMultivariateReturnEstimator(
            model="mistralai/Timestral-28M", horizon=21, device="cpu"
        )
        res1 = est_lazy.estimate(synthetic_market_data)
        res2 = est_lazy.estimate(synthetic_market_data)
        assert len(loaded_calls) == 1  # cached after first call
        assert loaded_calls[0] == ("mistralai/Timestral-28M", {"device": "cpu"})
        assert np.allclose(res1.values, res2.values)

    def test_end_to_end_pipeline_and_black_litterman_integration(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockTimestralForecaster(horizon=horizon)

        # 1. StrategyPipeline with TimestralMultivariateReturnEstimator
        pipeline = StrategyPipeline(
            allocator=MarkowitzAllocator(
                tickers=synthetic_market_data.assets,
                objective=ObjectiveFunction.MAX_SHARPE,
            ),
            returns_estimator=TimestralMultivariateReturnEstimator(
                model=model, horizon=horizon
            ),
            cov_estimator=LedoitWolfShrinkage(),
        )
        weights = pipeline.generate_weights(synthetic_market_data)
        assert np.isclose(weights.sum(), 1.0)
        assert list(weights.index) == synthetic_market_data.assets

        # 2. Black-Litterman pipeline with TimestralMultivariateViewGenerator
        view_gen = TimestralMultivariateViewGenerator(model=model, horizon=horizon)
        P, Q, Omega = view_gen.generate(synthetic_market_data)
        cov_matrix = LedoitWolfShrinkage().estimate(synthetic_market_data)
        market_caps = pd.Series(
            [1e12, 1.2e12, 0.9e12, 0.8e12], index=synthetic_market_data.assets
        )

        bl = BlackLittermanEstimator(tau=0.05, risk_aversion=2.5)
        mu_bl, cov_bl = bl.estimate(
            market_caps=market_caps,
            cov_matrix=cov_matrix,
            P=P,
            Q=Q,
            Omega=Omega,
        )
        assert isinstance(mu_bl, pd.Series)
        assert isinstance(cov_bl, pd.DataFrame)
        assert not mu_bl.isna().any()

        bl_pipeline = StrategyPipeline(
            allocator=MarkowitzAllocator(
                tickers=synthetic_market_data.assets,
                objective=ObjectiveFunction.MAX_SHARPE,
            ),
            returns_estimator=BlackLittermanReturnEstimator(
                P=P, Q=Q, Omega=Omega, tau=0.05
            ),
            cov_estimator=BlackLittermanCovarianceEstimator(
                P=P, Q=Q, Omega=Omega, tau=0.05
            ),
        )
        bl_weights = bl_pipeline.generate_weights(synthetic_market_data)
        assert np.isclose(bl_weights.sum(), 1.0)
