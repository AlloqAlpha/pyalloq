from __future__ import annotations

from typing import Any

import numpy as np
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_simulation.utils.imputation import FactorBackcastImputer
from sklearn.covariance import LedoitWolf


class ReactionDiffusionGenerator(BaseScenarioGenerator):
    """
    Stochastic Graph Reaction-Diffusion Contagion Scenario Generator.

    Inherits from `BaseScenarioGenerator` and outputs a 3D `ScenarioMarketData` tensor
    of shape `(M_paths, T_horizon, N_assets)`.

    Simulates multi-asset return and volatility trajectories where localized
    volatility shocks self-excite via FitzHugh-Nagumo bistable reaction dynamics and
    propagate across connected assets via the Precision Graph Laplacian L = I - D^{-1/2} W D^{-1/2}:

        dv_{m,t} = [ -D_v (v_{m,t} @ L^T) - kappa v_{m,t}
                     + gamma_react v_{m,t} (v_{m,t} - theta_crit)(1 - v_{m,t})
                     - beta_u u_{m,t} ] dt + vol_of_vol * dB^{(v)}_{m,t}
        du_{m,t} = [ -D_u (u_{m,t} @ L^T) + eps_u (v_{m,t} - 0.8 u_{m,t}) ] dt
        r_{m,t}  = mu + sigma_base * exp(v_{m,t}) * dB^{(r)}_{m,t}

    Incorporates the asymmetric leverage effect Corr(dB^{(r)}, dB^{(v)}) = leverage_rho <= 0
    so negative return shocks endogenously trigger local volatility spikes that diffuse across
    the asset network.
    """

    def __init__(
        self,
        diffusion_coeff: float = 0.35,
        inhibitor_diffusion: float = 0.10,
        mean_reversion: float = 0.20,
        reaction_gain: float = 0.35,
        critical_threshold: float = 0.25,
        inhibitor_coupling: float = 0.25,
        recovery_rate: float = 0.15,
        vol_of_vol: float = 0.18,
        leverage_rho: float = -0.45,
        fast_window: int = 10,
        implied_vol_feature: str = "implied_vol",
        annualization_factor: float = 252.0,
        initial_shock: dict[str, float] | None = None,
        dtype: Any = np.float32,
        impute_missing: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.diffusion_coeff = float(diffusion_coeff)
        self.inhibitor_diffusion = float(inhibitor_diffusion)
        self.mean_reversion = float(mean_reversion)
        self.reaction_gain = float(reaction_gain)
        self.critical_threshold = float(critical_threshold)
        self.inhibitor_coupling = float(inhibitor_coupling)
        self.recovery_rate = float(recovery_rate)
        self.vol_of_vol = float(vol_of_vol)
        self.leverage_rho = float(np.clip(leverage_rho, -0.99, 0.99))
        self.fast_window = int(fast_window)
        self.implied_vol_feature = implied_vol_feature
        self.annualization_factor = float(annualization_factor)
        self.initial_shock = initial_shock
        self.dtype = np.dtype(dtype)
        self.impute_missing = bool(impute_missing)
        self.random_state = random_state

        self.is_fitted = False
        self.log_returns = False
        self.assets: list[str] = []
        self.N: int = 0
        self.mean_vector: np.ndarray | None = None
        self.sigma_base: np.ndarray | None = None
        self.v0: np.ndarray | None = None
        self.u0: np.ndarray | None = None
        self.laplacian: np.ndarray | None = None
        self.corr_cholesky: np.ndarray | None = None
        self.initial_prices: np.ndarray | None = None
        self.fiedler_value_: float = 0.0
        self.mixing_time_: float = float("inf")
        self.fiedler_vector_: np.ndarray | None = None
        self.hawkes_branching_matrix_: np.ndarray | None = None
        self.hawkes_spectral_radius_: float = 0.0

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
        self.mean_vector = np.mean(ret_np, axis=0)

        sigma_slow = np.std(ret_np, axis=0, ddof=1)
        sigma_slow = np.where(sigma_slow > 1e-6, sigma_slow, 1e-3)
        self.sigma_base = sigma_slow

        fast_slice = ret_np[-min(len(ret_np), self.fast_window) :]
        sigma_fast = (
            np.std(fast_slice, axis=0, ddof=1) if len(fast_slice) >= 2 else sigma_slow
        )
        sigma_fast = np.where(sigma_fast > 1e-6, sigma_fast, sigma_slow)

        if self.implied_vol_feature in clean_data.features:
            iv_latest_ann = (
                clean_data.features[self.implied_vol_feature]
                .iloc[-1]
                .reindex(self.assets)
                .to_numpy(dtype=np.float64)
            )
            iv_daily = iv_latest_ann / np.sqrt(self.annualization_factor)
            valid = np.isfinite(iv_daily) & (iv_daily > 0.0)
            sigma_init = np.where(valid, 0.7 * iv_daily + 0.3 * sigma_fast, sigma_fast)
        else:
            sigma_init = sigma_fast

        v0 = np.log(np.clip(sigma_init, 1e-6, None) / sigma_slow)
        if self.initial_shock:
            for ticker, shock_val in self.initial_shock.items():
                if ticker in self.assets:
                    idx = self.assets.index(ticker)
                    v0[idx] += float(shock_val)

        self.v0 = np.clip(v0, -1.5, 2.5)
        self.u0 = 0.25 * np.maximum(0.0, self.v0)

        z_ret = (ret_np - self.mean_vector) / sigma_slow
        if len(z_ret) >= 3 and self.N > 1:
            try:
                lw = LedoitWolf().fit(z_ret)
                corr = lw.covariance_.astype(np.float64)
            except Exception:
                corr = np.cov(z_ret, rowvar=False)
        else:
            corr = np.eye(self.N, dtype=np.float64)

        d_inv = 1.0 / np.sqrt(np.clip(np.diag(corr), 1e-10, None))
        corr = corr * np.outer(d_inv, d_inv)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        theta = np.linalg.inv(corr + 1e-3 * np.eye(self.N, dtype=np.float64))
        theta_diag = np.sqrt(np.clip(np.diag(theta), 1e-10, None))
        partial_corr = -theta / np.outer(theta_diag, theta_diag)
        adjacency = np.maximum(0.0, partial_corr)
        np.fill_diagonal(adjacency, 0.0)

        deg = np.sum(adjacency, axis=1)
        deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
        norm_adj = adjacency * np.outer(deg_inv_sqrt, deg_inv_sqrt)
        laplacian = np.eye(self.N, dtype=np.float64) - norm_adj
        self.laplacian = 0.5 * (laplacian + laplacian.T)

        # Spectral Fiedler connectivity (lambda_2) & mixing time
        if self.N > 1:
            l_eigs, l_vecs = np.linalg.eigh(self.laplacian)
            order = np.argsort(l_eigs)
            self.fiedler_value_ = float(max(0.0, l_eigs[order[1]]))
            self.mixing_time_ = float(1.0 / max(self.fiedler_value_, 1e-6))
            f_vec = l_vecs[:, order[1]].astype(np.float64)
            self.fiedler_vector_ = -f_vec if f_vec[0] < 0 else f_vec
        else:
            self.fiedler_value_ = 0.0
            self.mixing_time_ = float("inf")
            self.fiedler_vector_ = np.ones(1, dtype=np.float64)

        # Lag-1 Wiener-Hopf Hawkes branching approximation on jump events
        jump_ind = (np.abs(z_ret) >= 1.75).astype(np.float64)
        if len(jump_ind) > 3 and self.N > 1:
            j_cent = jump_ind - np.mean(jump_ind, axis=0, keepdims=True)
            c0 = (j_cent.T @ j_cent) / len(j_cent) + 1e-2 * np.eye(self.N)
            c1 = (j_cent[1:].T @ j_cent[:-1]) / (len(j_cent) - 1)
            g_raw = np.maximum(0.0, np.linalg.solve(c0, c1.T).T)
            rad = float(np.max(np.abs(np.linalg.eigvals(g_raw))))
            if rad > 0.95 and rad > 1e-12:
                g_raw = g_raw * (0.95 / rad)
                rad = 0.95
            self.hawkes_branching_matrix_ = g_raw
            self.hawkes_spectral_radius_ = rad
        else:
            self.hawkes_branching_matrix_ = np.zeros((self.N, self.N), dtype=np.float64)
            self.hawkes_spectral_radius_ = 0.0

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.corr_cholesky = np.linalg.cholesky(corr_psd)
        self.is_fitted = True

    def generate(
        self,
        n_paths: int,
        horizon: int,
        initial_shock: dict[str, float] | None = None,
    ) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.mean_vector is None
            or self.sigma_base is None
            or self.v0 is None
            or self.u0 is None
            or self.laplacian is None
            or self.corr_cholesky is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "ReactionDiffusionGenerator needs to be fitted. Call .fit(data: MarketData) first."
            )

        v_curr = np.broadcast_to(self.v0, (n_paths, self.N)).astype(
            np.float64, copy=True
        )
        if initial_shock:
            for ticker, shock_val in initial_shock.items():
                if ticker in self.assets:
                    idx = self.assets.index(ticker)
                    v_curr[:, idx] += float(shock_val)

        u_curr = np.broadcast_to(self.u0, (n_paths, self.N)).astype(
            np.float64, copy=True
        )
        lap_T = self.laplacian.T
        chol_T = self.corr_cholesky.T
        rho = self.leverage_rho
        rho_orth = float(np.sqrt(max(0.0, 1.0 - rho**2)))

        R_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)
        IV_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)

        for t in range(horizon):
            sigma_t = self.sigma_base * np.exp(np.clip(v_curr, -1.5, 2.0))
            IV_sim[:, t, :] = sigma_t * np.sqrt(self.annualization_factor)

            z_ret = np.random.standard_normal(size=(n_paths, self.N)) @ chol_T
            z_indep = np.random.standard_normal(size=(n_paths, self.N))
            z_vol = rho * z_ret + rho_orth * z_indep

            R_sim[:, t, :] = self.mean_vector + sigma_t * z_ret

            diff_v = -self.diffusion_coeff * (v_curr @ lap_T)
            react_v = (
                self.reaction_gain
                * v_curr
                * (v_curr - self.critical_threshold)
                * (1.0 - v_curr)
            )
            dv = (
                diff_v
                - self.mean_reversion * v_curr
                + react_v
                - self.inhibitor_coupling * u_curr
                + self.vol_of_vol * z_vol
            )
            du = -self.inhibitor_diffusion * (u_curr @ lap_T) + self.recovery_rate * (
                v_curr - 0.8 * u_curr
            )

            v_curr = np.clip(v_curr + dv, -1.5, 2.0)
            u_curr = np.clip(u_curr + du, -1.5, 2.0)

        R_sim_out = R_sim.astype(self.dtype, copy=False)
        if self.log_returns:
            synthetic_prices = self.initial_prices * np.exp(
                np.cumsum(R_sim_out, axis=1)
            )
        else:
            synthetic_prices = self.initial_prices * np.cumprod(1.0 + R_sim_out, axis=1)

        synthetic_prices = synthetic_prices.astype(self.dtype, copy=False)
        sim_features, feat_cols = self._simulate_exogenous_features(R_sim_out)
        sim_features[self.implied_vol_feature] = IV_sim.astype(self.dtype, copy=False)
        feat_cols[self.implied_vol_feature] = list(self.assets)

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=R_sim_out,
            asset_names=self.assets,
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )
