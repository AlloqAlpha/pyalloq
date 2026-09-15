from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData

from pyalloq.dl.estimators.black_litterman.timesfm import (
    TimesFM3MultivariateViewGenerator,
    TimesFM3UnivariateViewGenerator,
)
from pyalloq.dl.estimators.returns.timesfm import (
    TimesFM3MultivariateReturnEstimator,
    TimesFM3UnivariateReturnEstimator,
)


class MockUnivariateTimesFMEvaluator:
    def __init__(self, horizon: int = 21) -> None:
        self.horizon = horizon

    def predict_batch(
        self, ts_list: list[np.ndarray], **kwargs
    ) -> list[SimpleNamespace]:
        outputs = []
        for ts in ts_list:
            last_val = ts[-1]
            # Forecast prices moving up slightly
            forecast = np.linspace(last_val * 1.01, last_val * 1.05, self.horizon)
            # Quantiles: shape (horizon, 9)
            quantiles = np.zeros((self.horizon, 9), dtype=np.float32)
            # p10 (idx 0), p50 (idx 4), p90 (idx 8)
            quantiles[:, 0] = forecast * 0.95
            quantiles[:, 4] = forecast
            quantiles[:, 8] = forecast * 1.05
            outputs.append(SimpleNamespace(forecast=forecast, quantiles=quantiles))
        return outputs


class MockMultivariateTimesFMEvaluator:
    def __init__(self, horizon: int = 21) -> None:
        self.horizon = horizon

    def predict_batch(
        self, contexts: list[np.ndarray], **kwargs
    ) -> list[SimpleNamespace]:
        # contexts[0] shape: (N_assets, T)
        n_assets = contexts[0].shape[0]
        last_vals = contexts[0][:, -1]

        # forecast shape: (N_assets, Horizon)
        forecast = np.zeros((n_assets, self.horizon), dtype=np.float32)
        for i in range(n_assets):
            forecast[i, :] = np.linspace(
                last_vals[i] * 1.01, last_vals[i] * 1.05, self.horizon
            )

        # quantiles shape: (N_assets, Horizon, 9)
        quantiles = np.zeros((n_assets, self.horizon, 9), dtype=np.float32)
        quantiles[:, :, 0] = forecast * 0.95
        quantiles[:, :, 4] = forecast
        quantiles[:, :, 8] = forecast * 1.05

        return [SimpleNamespace(forecast=forecast, quantiles=quantiles)]


@pytest.mark.dl
class TestTimesFMEstimators:
    def test_univariate_return_estimator(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockUnivariateTimesFMEvaluator(horizon=horizon)
        estimator = TimesFM3UnivariateReturnEstimator(model=model, horizon=horizon)

        returns = estimator.estimate(synthetic_market_data)
        assert isinstance(returns, pd.Series)
        assert list(returns.index) == synthetic_market_data.assets
        assert not returns.isna().any()
        # Returns should be positive since mock forecast went up
        assert (returns > 0).all()

    def test_multivariate_return_estimator(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockMultivariateTimesFMEvaluator(horizon=horizon)
        estimator = TimesFM3MultivariateReturnEstimator(
            model=model, horizon=horizon, annualize=True
        )

        returns = estimator.estimate(synthetic_market_data)
        assert isinstance(returns, pd.Series)
        assert list(returns.index) == synthetic_market_data.assets
        assert not returns.isna().any()
        assert (returns > 0).all()

    def test_univariate_view_generator(self, synthetic_market_data: MarketData) -> None:
        horizon = 21
        model = MockUnivariateTimesFMEvaluator(horizon=horizon)
        generator = TimesFM3UnivariateViewGenerator(tfm_model=model, horizon=horizon)

        P, Q, Omega = generator.generate(synthetic_market_data)
        n_assets = len(synthetic_market_data.assets)

        # P: identity
        assert isinstance(P, pd.DataFrame)
        assert np.allclose(P.values, np.eye(n_assets))

        # Q: expected returns series
        assert isinstance(Q, pd.Series)
        assert len(Q) == n_assets
        assert not Q.isna().any()

        # Omega: diagonal uncertainty matrix
        assert isinstance(Omega, pd.DataFrame)
        assert Omega.shape == (n_assets, n_assets)
        # Check off-diagonals are 0
        diag = np.diag(np.diag(Omega.values))
        assert np.allclose(Omega.values, diag)
        # Check diagonal elements are positive
        assert (np.diag(Omega.values) > 0).all()

    def test_multivariate_view_generator(
        self, synthetic_market_data: MarketData
    ) -> None:
        horizon = 21
        model = MockMultivariateTimesFMEvaluator(horizon=horizon)
        generator = TimesFM3MultivariateViewGenerator(model=model, horizon=horizon)

        P, Q, Omega = generator.generate(synthetic_market_data)
        n_assets = len(synthetic_market_data.assets)

        assert isinstance(P, pd.DataFrame)
        assert np.allclose(P.values, np.eye(n_assets))
        assert isinstance(Q, pd.Series)
        assert len(Q) == n_assets
        assert isinstance(Omega, pd.DataFrame)
        assert Omega.shape == (n_assets, n_assets)
        assert (np.diag(Omega.values) > 0).all()
