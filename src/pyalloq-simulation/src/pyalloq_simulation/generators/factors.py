from typing import Any
import numpy as np
from sklearn.decomposition import PCA

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class PCAFactorGenerator(BaseScenarioGenerator):
    """
    Factor-Based Monte Carlo Generator.
    Decomposes historical returns via PCA, simulates K latent factors,
    and reconstitutes N asset returns via static factor loadings.
    """

    def __init__(
        self,
        n_components: int = 5,
        dtype: Any = np.float32,
        impute_missing: bool = True,
    ) -> None:
        self.K = n_components
        self.dtype = np.dtype(dtype)
        self.impute_missing = impute_missing

        self.is_fitted = False
        self.log_returns: bool = False
        self.factor_means: np.ndarray | None = None
        self.factor_stds: np.ndarray | None = None
        self.loadings: np.ndarray | None = None
        self.asset_means: np.ndarray | None = None
        self.resid_stds: np.ndarray | None = None
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

        k_eff = min(self.K, self.N, len(returns))
        self.k_eff = k_eff
        self.pca = PCA(n_components=k_eff)

        ret_np = returns.to_numpy(dtype=np.float64)
        F_hist = self.pca.fit_transform(ret_np)
        self.loadings = self.pca.components_.astype(self.dtype)
        self.asset_means = self.pca.mean_.astype(self.dtype)

        R_reconstructed = self.pca.inverse_transform(F_hist)
        E_hist = ret_np - R_reconstructed

        self.factor_means = np.mean(F_hist, axis=0).astype(self.dtype)
        self.factor_stds = np.std(F_hist, axis=0).astype(self.dtype)
        self.resid_stds = np.std(E_hist, axis=0).astype(self.dtype)

        self.is_fitted = True

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.factor_means is None
            or self.factor_stds is None
            or self.loadings is None
            or self.asset_means is None
            or self.resid_stds is None
            or self.initial_prices is None
        ):
            raise ValueError("PCAFactorGenerator needs to be fitted. Call .fit() first")

        Z_factors = np.random.standard_normal(
            size=(n_paths, horizon, self.k_eff)
        ).astype(self.dtype)
        F_sim = self.factor_means + Z_factors * self.factor_stds

        Z_noise = np.random.standard_normal(size=(n_paths, horizon, self.N)).astype(
            self.dtype
        )
        E_sim = Z_noise * self.resid_stds

        R_sim = (self.asset_means + np.matmul(F_sim, self.loadings) + E_sim).astype(
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
