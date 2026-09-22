import numpy as np
from sklearn.decomposition import PCA

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData


class PCAFactorGenerator(BaseScenarioGenerator):
    """
    Factor-Based Monte Carlo Generator.
    Decomposes historical returns via PCA, simulates K latent factors,
    and reconstitutes N asset returns via static factor loadings.
    """

    def __init__(self, n_components: int = 5) -> None:
        self.K = n_components
        self.pca = PCA(n_components=self.K)

        self.is_fitted = False
        self.factor_means: np.ndarray | None = None
        self.factor_stds: np.ndarray | None = None
        self.loadings: np.ndarray | None = None
        self.resid_stds: np.ndarray | None = None

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        self.assets = data.assets
        self.N = len(self.assets)
        self.initial_prices = data.prices.iloc[-1].to_numpy(dtype=float)
        returns = data.get_returns(log_returns)

        F_hist = self.pca.fit_transform(returns)
        self.loadings = self.pca.components_

        R_reconstructed = F_hist @ self.loadings
        E_hist = returns.to_numpy(dtype=float) - R_reconstructed

        self.factor_means = np.mean(F_hist, axis=0)
        self.factor_stds = np.std(F_hist, axis=0)
        self.resid_stds = np.std(E_hist, axis=0)

        self.is_fitted = True

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.factor_means is None
            or self.factor_stds is None
            or self.loadings is None
            or self.resid_stds is None
            or self.initial_prices is None
        ):
            raise ValueError("FHS needs to be fitted. Call .fit() first")

        Z_factors = np.random.standard_normal(size=(n_paths, horizon, self.K))
        F_sim = self.factor_means + Z_factors * self.factor_stds

        Z_noise = np.random.standard_normal(size=(n_paths, horizon, self.N))
        E_sim = Z_noise * self.resid_stds

        R_sim = np.matmul(F_sim, self.loadings) + E_sim

        synthetic_prices = self.initial_prices * np.exp(np.cumsum(R_sim, axis=1))

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=R_sim,
            asset_names=self.assets,
        )
