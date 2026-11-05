import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline

from pyalloq.classical.estimators.black_litterman.bayesian import (
    BlackLittermanCovarianceEstimator,
    BlackLittermanEstimator,
    BlackLittermanReturnEstimator,
)
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator


class TestBlackLitterman:
    def test_black_litterman_views_integration(self) -> None:
        assets = ["AAPL", "MSFT", "GOOGL"]
        market_caps = pd.Series([1000.0, 2000.0, 1500.0], index=assets)

        cov_matrix = pd.DataFrame(
            [
                [0.04, 0.01, 0.015],
                [0.01, 0.03, 0.012],
                [0.015, 0.012, 0.035],
            ],
            index=assets,
            columns=assets,
        )

        # View: AAPL will outperform MSFT by 2% (0.02)
        P = pd.DataFrame([[1.0, -1.0, 0.0]], columns=assets)
        Q = pd.DataFrame([0.02])

        bl = BlackLittermanEstimator(tau=0.05, risk_aversion=2.5)
        post_mu, post_cov = bl.estimate(
            market_caps=market_caps, cov_matrix=cov_matrix, P=P, Q=Q
        )

        # 1. Output shapes and types
        assert isinstance(post_mu, pd.Series)
        assert isinstance(post_cov, pd.DataFrame)
        assert list(post_mu.index) == assets
        assert list(post_cov.index) == assets
        assert list(post_cov.columns) == assets

        # 2. Invariant: Posterior covariance must be symmetric and positive definite
        np.testing.assert_allclose(post_cov.values, post_cov.values.T, atol=1e-7)
        eigenvalues = np.linalg.eigvalsh(post_cov.values)
        assert np.all(eigenvalues > 0), "Posterior covariance must be positive definite"

        # 3. Directional check: AAPL return should be higher than without the view relative to MSFT
        assert post_mu["AAPL"] > post_mu["MSFT"]

    def test_black_litterman_pipeline_adapters(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        P = pd.DataFrame([[1.0, -1.0, 0.0, 0.0]], columns=sample_assets)
        Q = pd.Series([0.03])
        synthetic_market_data.cross_sectional = pd.DataFrame(
            {"market_cap": [1e9, 1e9, 1e9, 1e9]}, index=sample_assets
        )

        ret_est = BlackLittermanReturnEstimator(P=P, Q=Q)
        cov_est = BlackLittermanCovarianceEstimator(P=P, Q=Q)
        allocator = MarkowitzAllocator(tickers=sample_assets)
        pipeline = StrategyPipeline(
            allocator=allocator, returns_estimator=ret_est, cov_estimator=cov_est
        )

        weights = pipeline.generate_weights(synthetic_market_data)
        assert isinstance(weights, pd.Series)
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
