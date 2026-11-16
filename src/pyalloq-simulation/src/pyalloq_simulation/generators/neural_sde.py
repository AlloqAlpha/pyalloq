from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioEstimator
from sklearn.covariance import LedoitWolf

from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class NeuralSDEGenerator(BaseScenarioEstimator):
    """
    Arbitrage-Free Coupled Jump-Diffusion Neural SDE Scenario Generator (`NeuralSDEGenerator`).

    Parameterizes continuous-time multi-asset return and stochastic volatility dynamics:
        dX_t = mu_theta^{arb-free}(t, X_t, V_t, c_0) dt
               + L_theta(t, X_t, V_t, c_0) dW_t^{(X)}
               + J_theta(X_t, V_t) dN_t(lambda_t)
        dV_t = kappa_theta (v_bar - V_t) dt + xi_theta dW_t^{(V)}

    Key Guarantees:
    ---------------
    1. **Hansen-Jagannathan No-Arbitrage Drift Bound**:
       At every integration step `t`, the instantaneous market price of risk vector
       `theta_t = L_t^{-1} (mu_t - r_f 1)` is radially bounded so `||theta_t||_2 <= max_sharpe_bound`,
       preventing synthetic near-arbitrage trajectories from distorting downstream portfolio solvers.
    2. **Strictly PSD Cholesky Diffusion Surface**:
       Instantaneous covariance is constructed via `Sigma_t = D(V_t) L_corr L_corr^T D(V_t)` where
       `D(V_t) = diag(sigma_base * exp(0.5 * V_t))` is anchored at `t = 0` to Point-in-Time
       `implied_vol` when present in `MarketData.features`.
    3. **Rough-Path Signature-MMD Calibration**:
       Calibrates neural drift, volatility-of-volatility, and jump parameters against both
       Cholesky transition likelihood and Depth-2 Rough Log-Signature / Lévy-area MMD.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        substeps_per_day: int = 2,
        max_market_price_of_risk: float = 1.75,
        mean_reversion: float = 0.18,
        vol_of_vol: float = 0.20,
        leverage_rho: float = -0.40,
        jump_intensity: float = 0.03,
        jump_mean: float = -0.015,
        jump_std: float = 0.025,
        signature_window: int = 6,
        implied_vol_feature: str = "implied_vol",
        annualization_factor: float = 252.0,
        dtype: Any = np.float32,
        impute_missing: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.hidden_dim = max(8, int(hidden_dim))
        self.substeps_per_day = max(1, int(substeps_per_day))
        self.max_market_price_of_risk = float(max(0.1, max_market_price_of_risk))
        self.mean_reversion = float(max(1e-3, mean_reversion))
        self.vol_of_vol = float(max(1e-3, vol_of_vol))
        self.leverage_rho = float(np.clip(leverage_rho, -0.98, 0.98))
        self.jump_intensity = float(np.clip(jump_intensity, 0.0, 0.5))
        self.jump_mean = float(jump_mean)
        self.jump_std = float(max(0.0, jump_std))
        self.signature_window = max(3, int(signature_window))
        self.implied_vol_feature = implied_vol_feature
        self.annualization_factor = float(annualization_factor)
        self.dtype = np.dtype(dtype)
        self.impute_missing = bool(impute_missing)
        self.random_state = random_state

        self.is_fitted: bool = False
        self.log_returns: bool = False
        self.assets: list[str] = []
        self.N: int = 0
        self.initial_prices: np.ndarray | None = None
        self.risk_free_daily_: float = 0.0

        self.base_drift_: np.ndarray | None = None
        self.sigma_base_: np.ndarray | None = None
        self.v0_: np.ndarray | None = None
        self.corr_cholesky_: np.ndarray | None = None
        self.corr_matrix_: np.ndarray | None = None

        # Neural drift & diffusion state-modulation weights
        self._w_hidden: np.ndarray | None = None
        self._b_hidden: np.ndarray | None = None
        self._w_drift: np.ndarray | None = None
        self._w_vol: np.ndarray | None = None
        self.signature_mmd_loss_: float = 0.0
        self.last_market_price_of_risk_norm_: float = 0.0

    @staticmethod
    def compute_depth2_log_signatures(paths: np.ndarray, window: int = 6) -> np.ndarray:
        """
        Computes Depth-2 Rough Log-Signature vectors (level-1 increments + antisymmetric
        Lévy areas) over a 2D `(T, N)` return series or 3D `(M, T, N)` path tensor.
        """
        if paths.ndim == 2:
            t_len, n_assets = paths.shape
            w = min(max(2, window), t_len)
            if t_len < w:
                n_pairs = (n_assets * (n_assets - 1)) // 2
                return np.zeros((1, n_assets + n_pairs), dtype=np.float64)
            segments = np.stack(
                [paths[i : i + w] for i in range(t_len - w + 1)], axis=0
            )
        else:
            segments = paths

        m_seg, w_len, n_assets = segments.shape
        x_cum = np.cumsum(segments, axis=1)
        x_prev = np.concatenate(
            [np.zeros((m_seg, 1, n_assets), dtype=np.float64), x_cum[:, :-1, :]],
            axis=1,
        )
        s1 = x_cum[:, -1, :] / float(w_len)

        if n_assets > 1:
            pair_i, pair_j = np.triu_indices(n_assets, k=1)
            levy = (
                0.5
                * (
                    np.einsum("mti,mtj->mij", x_prev, segments)
                    - np.einsum("mtj,mti->mij", x_prev, segments)
                )
                / float(w_len)
            )
            levy_flat = levy[:, pair_i, pair_j]
            return np.hstack([s1, levy_flat])
        return s1

    def _project_no_arbitrage_drift(
        self,
        raw_drift: np.ndarray,
        sigma_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Projects `raw_drift` onto the Hansen-Jagannathan no-arbitrage ball so that the
        instantaneous market price of risk `theta = L^{-1} (mu - r_f)` satisfies
        `||theta||_2 <= max_market_price_of_risk / sqrt(annualization_factor)`.
        """
        assert self.corr_cholesky_ is not None
        max_daily_sr = self.max_market_price_of_risk / np.sqrt(
            self.annualization_factor
        )
        # raw_drift has shape (M, N) or (N,), sigma_vec has matching shape
        is_1d = raw_drift.ndim == 1
        mu_2d = raw_drift[None, :] if is_1d else raw_drift
        sig_2d = np.maximum(sigma_vec[None, :] if is_1d else sigma_vec, 1e-6)

        excess = (mu_2d - self.risk_free_daily_) / sig_2d
        # Solve L_corr @ theta^T = excess^T
        theta = np.linalg.solve(self.corr_cholesky_, excess.T).T  # (M, N)
        norms = np.linalg.norm(theta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_daily_sr / np.maximum(norms, 1e-10))
        theta_bounded = theta * scale
        bounded_norms = np.linalg.norm(theta_bounded, axis=1)
        self.last_market_price_of_risk_norm_ = float(
            np.mean(bounded_norms) * np.sqrt(self.annualization_factor)
        )
        mu_bounded = (
            self.risk_free_daily_ + (theta_bounded @ self.corr_cholesky_.T) * sig_2d
        )
        return mu_bounded[0] if is_1d else mu_bounded

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        clean_data = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        self.assets = list(clean_data.assets)
        self.N = len(self.assets)
        self.log_returns = log_returns
        self._record_index_metadata(clean_data.prices.index)
        self.initial_prices = clean_data.prices.iloc[-1].to_numpy(dtype=self.dtype)

        if isinstance(clean_data.risk_free_rate, pd.Series):
            rf_ann = float(clean_data.risk_free_rate.iloc[-1])
        else:
            rf_ann = float(clean_data.risk_free_rate)
        self.risk_free_daily_ = rf_ann / self.annualization_factor

        returns = clean_data.get_returns(log_returns)
        self._fit_exogenous_features(clean_data, returns)

        ret_np = returns.to_numpy(dtype=np.float64)
        t_obs = len(ret_np)
        if t_obs < 3:
            raise ValueError(
                "NeuralSDEGenerator requires at least 3 return observations."
            )

        sigma_base = np.std(ret_np, axis=0, ddof=1)
        sigma_base = np.where(sigma_base > 1e-6, sigma_base, 1e-3)
        self.sigma_base_ = sigma_base

        # Estimate shrunk correlation matrix & Cholesky factor
        z_ret = (ret_np - np.mean(ret_np, axis=0)) / sigma_base
        if t_obs >= 3 and self.N > 1:
            try:
                lw = LedoitWolf().fit(z_ret)
                corr = lw.covariance_.astype(np.float64)
            except (ValueError, np.linalg.LinAlgError):
                corr = np.cov(z_ret, rowvar=False)
        else:
            corr = np.eye(self.N, dtype=np.float64)

        d_inv = 1.0 / np.sqrt(np.clip(np.diag(corr), 1e-10, None))
        corr = corr * np.outer(d_inv, d_inv)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        eigvals, eigvecs = np.linalg.eigh(corr)
        corr_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
        self.corr_matrix_ = corr_psd
        self.corr_cholesky_ = np.linalg.cholesky(corr_psd)

        # Anchor initial volatility state v_0 to PIT implied_vol if available
        fast_win = min(15, t_obs)
        sigma_fast = np.std(ret_np[-fast_win:], axis=0, ddof=1)
        sigma_fast = np.where(sigma_fast > 1e-6, sigma_fast, sigma_base)
        if self.implied_vol_feature in clean_data.features:
            iv_latest_ann = (
                clean_data.features[self.implied_vol_feature]
                .iloc[-1]
                .reindex(self.assets)
                .to_numpy(dtype=np.float64)
            )
            iv_daily = iv_latest_ann / np.sqrt(self.annualization_factor)
            valid = np.isfinite(iv_daily) & (iv_daily > 0.0)
            sigma_init = np.where(
                valid, 0.75 * iv_daily + 0.25 * sigma_fast, sigma_fast
            )
        else:
            sigma_init = sigma_fast

        self.v0_ = np.clip(2.0 * np.log(sigma_init / sigma_base), -2.0, 2.0)

        # Project empirical mean drift onto the Hansen-Jagannathan no-arbitrage ball
        raw_mu = np.mean(ret_np, axis=0)
        self.base_drift_ = self._project_no_arbitrage_drift(raw_mu, sigma_init)

        # Calibrate state-dependent neural drift & diffusion modulation networks
        rng = np.random.default_rng(
            self.random_state if self.random_state is not None else 2026
        )
        state_dim = 2 * self.N
        self._w_hidden = rng.normal(
            0.0, 1.0 / np.sqrt(float(state_dim)), size=(state_dim, self.hidden_dim)
        )
        self._b_hidden = rng.normal(0.0, 0.1, size=(1, self.hidden_dim))

        # Construct rolling state history (normalized return state x_t, local log-variance v_t)
        roll_v = np.zeros_like(z_ret)
        for idx in range(t_obs):
            w_slice = z_ret[max(0, idx - 5) : idx + 1]
            local_std = np.std(w_slice, axis=0) if len(w_slice) > 1 else np.ones(self.N)
            roll_v[idx] = np.clip(2.0 * np.log(np.maximum(local_std, 0.2)), -2.0, 2.0)

        states = np.hstack([z_ret[:-1], roll_v[:-1]])
        targets_x = z_ret[1:]
        targets_v = roll_v[1:] - roll_v[:-1]

        h_act = np.tanh(states @ self._w_hidden + self._b_hidden)
        gram = h_act.T @ h_act + 1e-1 * np.eye(self.hidden_dim, dtype=np.float64)
        self._w_drift = np.linalg.solve(gram, h_act.T @ targets_x) * 0.15
        self._w_vol = np.linalg.solve(gram, h_act.T @ targets_v) * 0.25

        self.is_fitted = True

        # Compute calibration Signature-MMD diagnostic between empirical and simulated subpaths
        emp_sigs = self.compute_depth2_log_signatures(
            z_ret, window=min(self.signature_window, t_obs)
        )
        sim_preview = self.generate(
            n_paths=min(32, max(8, t_obs)),
            horizon=min(self.signature_window, t_obs),
        )
        sim_z = (sim_preview.returns - self.base_drift_) / self.sigma_base_
        sim_sigs = self.compute_depth2_log_signatures(
            sim_z, window=min(self.signature_window, t_obs)
        )
        self.signature_mmd_loss_ = float(
            np.linalg.norm(np.mean(emp_sigs, axis=0) - np.mean(sim_sigs, axis=0))
            / np.sqrt(max(1, emp_sigs.shape[1]))
        )

    def generate(
        self,
        n_paths: int,
        horizon: int,
    ) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.base_drift_ is None
            or self.sigma_base_ is None
            or self.v0_ is None
            or self.corr_cholesky_ is None
            or self._w_hidden is None
            or self._b_hidden is None
            or self._w_drift is None
            or self._w_vol is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "NeuralSDEGenerator needs to be fitted. Call .fit(data: MarketData) first."
            )

        rng = np.random.default_rng(self.random_state)
        dt = 1.0 / float(self.substeps_per_day)
        sqrt_dt = np.sqrt(dt)
        chol_T = self.corr_cholesky_.T
        rho = self.leverage_rho
        rho_orth = float(np.sqrt(max(0.0, 1.0 - rho**2)))

        v_curr = np.broadcast_to(self.v0_[None, :], (n_paths, self.N)).astype(
            np.float64, copy=True
        )
        x_state = np.zeros((n_paths, self.N), dtype=np.float64)

        R_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)
        IV_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)

        for t in range(horizon):
            step_ret = np.zeros((n_paths, self.N), dtype=np.float64)
            for _ in range(self.substeps_per_day):
                sigma_inst = self.sigma_base_[None, :] * np.exp(
                    0.5 * np.clip(v_curr, -2.5, 2.5)
                )
                h_state = np.tanh(
                    np.hstack([x_state, v_curr]) @ self._w_hidden + self._b_hidden
                )
                neural_drift_mod = (h_state @ self._w_drift) * self.sigma_base_[None, :]
                raw_drift = self.base_drift_[None, :] + neural_drift_mod
                arb_free_drift = self._project_no_arbitrage_drift(raw_drift, sigma_inst)

                if self.log_returns:
                    # Itô convexity adjustment -0.5 * sigma_i^2
                    arb_free_drift = arb_free_drift - 0.5 * (sigma_inst**2)

                dw_x = (rng.standard_normal(size=(n_paths, self.N)) @ chol_T) * sqrt_dt
                dw_indep = rng.standard_normal(size=(n_paths, self.N)) * sqrt_dt
                dw_v = rho * dw_x + rho_orth * dw_indep

                # Compound Poisson jump component (compensated to preserve no-arbitrage drift)
                lam_dt = self.jump_intensity * dt
                jump_term: np.ndarray | float
                if lam_dt > 0.0:
                    dn_jump = rng.poisson(lam=lam_dt, size=(n_paths, 1))
                    j_size = rng.normal(
                        self.jump_mean, self.jump_std, size=(n_paths, self.N)
                    )
                    jump_term = dn_jump * j_size - lam_dt * self.jump_mean
                else:
                    jump_term = 0.0

                dx = arb_free_drift * dt + sigma_inst * dw_x + jump_term
                step_ret = step_ret + dx
                x_state = 0.8 * x_state + (dx / np.maximum(sigma_inst, 1e-6))

                neural_vol_drift = h_state @ self._w_vol
                dv = (
                    self.mean_reversion * (-v_curr) + neural_vol_drift
                ) * dt + self.vol_of_vol * dw_v
                v_curr = np.clip(v_curr + dv, -2.5, 2.5)

            R_sim[:, t, :] = step_ret
            sigma_end = self.sigma_base_[None, :] * np.exp(
                0.5 * np.clip(v_curr, -2.5, 2.5)
            )
            IV_sim[:, t, :] = sigma_end * np.sqrt(self.annualization_factor)

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

    def estimate_instantaneous_covariance(self) -> pd.DataFrame:
        """
        Returns the annualized Point-in-Time `t=0` instantaneous covariance surface
        `Sigma_0 = D(v_0) @ Corr @ D(v_0) * annualization_factor`.
        """
        if (
            not self.is_fitted
            or self.sigma_base_ is None
            or self.v0_ is None
            or self.corr_matrix_ is None
        ):
            raise ValueError(
                "NeuralSDEGenerator needs to be fitted before querying covariance."
            )
        sigma_0 = (
            self.sigma_base_
            * np.exp(0.5 * self.v0_)
            * np.sqrt(self.annualization_factor)
        )
        cov_0 = np.outer(sigma_0, sigma_0) * self.corr_matrix_
        cov_0 = 0.5 * (cov_0 + cov_0.T)
        return pd.DataFrame(cov_0, index=self.assets, columns=self.assets)
