import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.covariance.ewma import EWMACovariance
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.covariance.random_matrix_theory import (
    RandomMatrixTheoryEstimator,
)
from pyalloq.classical.estimators.covariance.semi_covariance import SemiCovariance


class TestCovarianceEstimators:
    @pytest.mark.parametrize(
        "estimator_cls, kwargs",
        [
            (EmpiricalCovariance, {"annualization_factor": 252.0}),
            (EWMACovariance, {"span": 30, "annualization_factor": 252.0}),
            (LedoitWolfShrinkage, {"annualization_factor": 252.0}),
            (RandomMatrixTheoryEstimator, {"denoise": True, "detone": False}),
            (SemiCovariance, {"benchmark_return": 0.0, "annualization_factor": 252.0}),
        ],
    )
    def test_covariance_mathematical_invariants(
        self,
        synthetic_market_data: MarketData,
        sample_assets: list[str],
        estimator_cls,
        kwargs,
    ) -> None:
        estimator = estimator_cls(**kwargs)
        cov = estimator.estimate(synthetic_market_data)

        # 1. Output type & dimensions
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (len(sample_assets), len(sample_assets))
        assert list(cov.columns) == sample_assets
        assert list(cov.index) == sample_assets

        # 2. Symmetry invariant: Sigma == Sigma.T
        np.testing.assert_allclose(
            cov.values, cov.values.T, atol=1e-6, err_msg="Covariance must be symmetric"
        )

        # 3. Positive Semi-Definite (PSD) invariant: eigenvalues >= -1e-6
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(
            eigenvalues >= -1e-6
        ), f"Eigenvalues must be non-negative, got {eigenvalues}"

        # 4. Variance invariant: diagonal elements must be non-negative
        variances = np.diag(cov.values)
        assert np.all(
            variances >= 0.0
        ), f"Variances must be non-negative, got {variances}"

    def test_annualization_scaling(self, synthetic_market_data: MarketData) -> None:
        cov_daily = EmpiricalCovariance(annualization_factor=1.0).estimate(
            synthetic_market_data
        )
        cov_annual = EmpiricalCovariance(annualization_factor=252.0).estimate(
            synthetic_market_data
        )

        np.testing.assert_allclose(
            cov_annual.values, cov_daily.values * 252.0, rtol=1e-5
        )

    def test_collinear_market_data_resilience(
        self, collinear_market_data: MarketData
    ) -> None:
        # Ledoit-Wolf should regularize and produce strictly positive definite covariance even on collinear data
        lw = LedoitWolfShrinkage().estimate(collinear_market_data)
        eigenvalues = np.linalg.eigvalsh(lw.values)
        assert np.all(
            eigenvalues > 0
        ), "Ledoit-Wolf must produce positive definite matrix on collinear data"

    def test_semi_covariance_downside_only(self) -> None:
        # If returns are strictly positive, semi-covariance should be 0
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        # Monotonically increasing prices -> positive returns
        prices = pd.DataFrame(
            {
                "A": [100.0 * (1.01**i) for i in range(10)],
                "B": [100.0 * (1.02**i) for i in range(10)],
            },
            index=dates,
        )
        md = MarketData(prices=prices)
        semi_cov = SemiCovariance(benchmark_return=0.0).estimate(md)
        np.testing.assert_allclose(semi_cov.values, np.zeros((2, 2)), atol=1e-7)
