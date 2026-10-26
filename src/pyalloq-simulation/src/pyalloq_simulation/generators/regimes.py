from typing import Any
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class GMMRegimeGenerator(BaseScenarioGenerator):
    """
    Gaussian Mixture Model (GMM) Markov Regime-Switching Monte Carlo Generator.

    1. Fits a K-component Gaussian Mixture Model on historical multivariate returns
       to identify latent macro regimes (e.g., Bull, Bear, High-Vol Crisis).
    2. Estimates the empirical K x K Markov state transition matrix P from historical
       regime transitions.
    3. Simulates dynamic regime switching along each path m and samples return vectors
       from the active regime's mean and covariance matrix.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        random_state: int | None = 42,
        dtype: Any = np.float32,
        impute_missing: bool = True,
    ) -> None:
        if n_regimes < 1:
            raise ValueError("`n_regimes` must be >= 1.")
        self.K = n_regimes
        self.random_state = random_state
        self.dtype = np.dtype(dtype)
        self.impute_missing = impute_missing

        self.is_fitted = False
        self.log_returns: bool = False
        self.assets: list[str] = []
        self.N: int = 0
        self.regime_means: np.ndarray | None = None
        self.regime_chols: np.ndarray | None = None
        self.transition_matrix: np.ndarray | None = None
        self.initial_state_probs: np.ndarray | None = None
        self.initial_prices: np.ndarray | None = None
        self.gmm: GaussianMixture | None = None
        self.regime_order: np.ndarray | None = None
        self.historical_regime_probs: pd.DataFrame | None = None
        self.regime_labels: list[str] = []

    def _default_regime_labels(self, k_eff: int) -> list[str]:
        if k_eff == 1:
            return ["Regime 0 (Normal)"]
        if k_eff == 2:
            return ["Regime 0 (Bull / Low-Vol)", "Regime 1 (Bear / High-Vol)"]
        if k_eff == 3:
            return [
                "Regime 0 (Bull / Low-Vol)",
                "Regime 1 (Neutral / Transition)",
                "Regime 2 (Crisis / High-Vol)",
            ]
        return [f"Regime {k}" for k in range(k_eff)]

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
        k_eff = min(self.K, max(1, len(ret_np) // 5))
        self.k_eff = k_eff

        gmm = GaussianMixture(
            n_components=k_eff,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=self.random_state,
        )
        gmm.fit(ret_np)
        self.gmm = gmm

        # Sort regimes from lowest average variance (Bull/Calm) to highest variance (Bear/Crisis)
        avg_vars = np.array(
            [np.trace(gmm.covariances_[k]) / max(1, self.N) for k in range(k_eff)]
        )
        order = np.argsort(avg_vars)
        self.regime_order = order
        inv_order = np.argsort(order)

        self.regime_means = gmm.means_[order].astype(self.dtype)  # (K, N)
        chols = np.zeros((k_eff, self.N, self.N), dtype=self.dtype)
        for idx, k in enumerate(order):
            cov_k = gmm.covariances_[k].astype(np.float64)
            eigvals, eigvecs = np.linalg.eigh(cov_k)
            eigvals = np.maximum(eigvals, 1e-8)
            cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            chols[idx] = np.linalg.cholesky(cov_pd).astype(self.dtype)
        self.regime_chols = chols

        # Estimate empirical Markov transition matrix from decoded regime sequence
        raw_states = gmm.predict(ret_np)
        states = inv_order[raw_states]
        trans_counts = np.full((k_eff, k_eff), 1e-2, dtype=np.float64)
        for t in range(len(states) - 1):
            trans_counts[states[t], states[t + 1]] += 1.0
        self.transition_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        # Store historical regime posterior probabilities
        raw_probs = gmm.predict_proba(ret_np)[:, order]
        self.regime_labels = self._default_regime_labels(k_eff)
        self.historical_regime_probs = pd.DataFrame(
            raw_probs, index=returns.index, columns=self.regime_labels
        )

        # Seed t=0 state distribution from the posterior probabilities of the final observation
        last_probs = raw_probs[-1]
        self.initial_state_probs = last_probs / np.sum(last_probs)
        self.is_fitted = True

    def predict_regime_probs(self, data: MarketData) -> pd.DataFrame:
        """
        Decodes real-time posterior regime probabilities p_k(t) on a MarketData instance.
        """
        if not self.is_fitted or self.gmm is None or self.regime_order is None:
            raise ValueError(
                "GMMRegimeGenerator needs to be fitted. Call .fit() first."
            )
        clean_data = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        returns = clean_data.get_returns(self.log_returns)
        ret_np = returns.to_numpy(dtype=np.float64)
        probs = self.gmm.predict_proba(ret_np)[:, self.regime_order]
        return pd.DataFrame(probs, index=returns.index, columns=self.regime_labels)

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.regime_means is None
            or self.regime_chols is None
            or self.transition_matrix is None
            or self.initial_state_probs is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "GMMRegimeGenerator needs to be fitted. Call .fit() first."
            )

        k_eff = self.k_eff
        states_sim = np.zeros((n_paths, horizon), dtype=np.int64)
        states_sim[:, 0] = np.random.choice(
            k_eff, size=n_paths, p=self.initial_state_probs
        )

        # Evolve Markov chains across paths
        for t in range(1, horizon):
            prev_states = states_sim[:, t - 1]
            u = np.random.random(size=n_paths)
            cum_trans = np.cumsum(self.transition_matrix[prev_states], axis=1)
            states_sim[:, t] = np.clip(
                (u[:, None] > cum_trans).sum(axis=1), 0, k_eff - 1
            )

        # Sample regime-conditional multivariate normal returns
        Z = np.random.standard_normal(size=(n_paths, horizon, self.N)).astype(
            self.dtype
        )
        means_t = self.regime_means[states_sim]  # (M, T, N)
        chols_t = self.regime_chols[states_sim]  # (M, T, N, N)
        # Batched matrix-vector product: L_{m,t} @ Z_{m,t}
        correlated_shocks = np.einsum("mtij,mtj->mti", chols_t, Z)
        R_sim = (means_t + correlated_shocks).astype(self.dtype, copy=False)

        if self.log_returns:
            synthetic_prices = self.initial_prices * np.exp(np.cumsum(R_sim, axis=1))
        else:
            synthetic_prices = self.initial_prices * np.cumprod(1.0 + R_sim, axis=1)

        synthetic_prices = synthetic_prices.astype(self.dtype, copy=False)
        sim_features, feat_cols = self._simulate_exogenous_features(R_sim)
        sim_features["regime_states"] = states_sim[:, :, None].astype(self.dtype)
        feat_cols["regime_states"] = ["regime_state"]

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=R_sim,
            asset_names=self.assets,
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )
