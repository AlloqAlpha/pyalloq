from typing import Any
import numpy as np
from sklearn.covariance import LedoitWolf

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class GaussianGenerator(BaseScenarioGenerator):
    """
    Multivariate Normal Monte Carlo Baseline Simulator.

    Estimates Ledoit-Wolf shrunk covariance on historical returns and generates
    correlated paths via Cholesky factorization Sigma = L @ L.T.
    """

    def __init__(
        self,
        dtype: Any = np.float32,
        impute_missing: bool = True,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.impute_missing = impute_missing
        self.is_fitted = False
        self.log_returns: bool = False
        self.assets: list[str] = []
        self.N: int = 0
        self.mean_vector: np.ndarray | None = None
        self.cov_matrix: np.ndarray | None = None
        self.cholesky_L: np.ndarray | None = None
        self.initial_prices: np.ndarray | None = None

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        clean_data = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        self.assets = list(clean_data.assets)
        self.N = len(self.assets)
        self.log_returns = log_returns
        self._record_index_metadata(clean_data.prices.index)
        self.initial_prices = clean_data.prices.iloc[-1].to_numpy(dtype=self.dtype)

        returns = clean_data.get_returns(log_returns)
        self._fit_exogenous_features(clean_data, returns)

        ret_np = returns.to_numpy(dtype=np.float64)
        self.mean_vector = np.mean(ret_np, axis=0).astype(self.dtype)

        lw = LedoitWolf().fit(ret_np)
        cov = lw.covariance_.astype(np.float64)

        # Ensure strict positive-definiteness for Cholesky factorization
        jitter = 1e-8
        for _ in range(5):
            try:
                L = np.linalg.cholesky(cov)
                break
            except np.linalg.LinAlgError:
                cov = cov + jitter * np.eye(self.N, dtype=np.float64)
                jitter *= 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        self.cov_matrix = cov.astype(self.dtype)
        self.cholesky_L = L.astype(self.dtype)
        self.is_fitted = True

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.mean_vector is None
            or self.cholesky_L is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "GaussianGenerator needs to be fitted. Call .fit(data: MarketData) first"
            )

        Z = np.random.standard_normal(size=(n_paths, horizon, self.N)).astype(
            self.dtype
        )
        # Correlate innovations: Z @ L.T has covariance L @ L.T = Sigma
        R_sim = (self.mean_vector + np.matmul(Z, self.cholesky_L.T)).astype(
            self.dtype, copy=False
        )

        if self.log_returns:
            synthetic_prices = self.initial_prices * np.exp(np.cumsum(R_sim, axis=1))
        else:
            synthetic_prices = self.initial_prices * np.cumprod(1.0 + R_sim, axis=1)

        synthetic_prices = synthetic_prices.astype(self.dtype, copy=False)
        sim_features, feat_cols = self._simulate_exogenous_features(R_sim)

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=R_sim,
            asset_names=self.assets,
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )
