from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseCovarianceEstimator
from sklearn.covariance import LedoitWolf

from pyalloq.classical.estimators.covariance.hawkes_spectral import (
    SpectralHawkesDiagnostics,
    compute_spectral_hawkes_diagnostics,
)


class ReactionDiffusionCovarianceEstimator(BaseCovarianceEstimator):
    """
    Graph Reaction-Diffusion Volatility & Contagion Covariance Estimator.

    Models cross-asset volatility spillover and non-linear self-excitation on a
    dynamic Financial Graph G_t = (V, E, W_t) derived from the shrunk precision
    (partial correlation) matrix Theta_t = Sigma_LW^{-1}:

        dv/dt = -D_v * L_t @ v - kappa * v + gamma_react * v * (v - theta_crit) * (1 - v) - beta_u * u
        du/dt = -D_u * L_t @ u + eps_u * (v - 0.8 * u)

    where:
    - `v_t in R^N` is the normalized log-volatility excitation state (anchored at t=0 by
      `data.features[implied_vol_feature]` when present, or fast-window realized volatility).
    - `u_t in R^N` is the slow liquidity-recovery / inhibitor field.
    - `L_t = I - D_t^{-1/2} W_t D_t^{-1/2}` is the symmetric normalized Graph Laplacian
      whose spectrum lies in [0, 2], propagating localized shocks across connected assets.
    """

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 60,
        horizon_steps: int = 5,
        diffusion_coeff: float = 0.35,
        inhibitor_diffusion: float = 0.10,
        mean_reversion: float = 0.20,
        reaction_gain: float = 0.40,
        critical_threshold: float = 0.25,
        inhibitor_coupling: float = 0.25,
        recovery_rate: float = 0.15,
        implied_vol_feature: str = "implied_vol",
        annualization_factor: float = 252.0,
    ) -> None:
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.horizon_steps = int(horizon_steps)
        self.diffusion_coeff = float(diffusion_coeff)
        self.inhibitor_diffusion = float(inhibitor_diffusion)
        self.mean_reversion = float(mean_reversion)
        self.reaction_gain = float(reaction_gain)
        self.critical_threshold = float(critical_threshold)
        self.inhibitor_coupling = float(inhibitor_coupling)
        self.recovery_rate = float(recovery_rate)
        self.implied_vol_feature = implied_vol_feature
        self.annualization_factor = float(annualization_factor)

        self.last_laplacian_: np.ndarray | None = None
        self.last_adjacency_: np.ndarray | None = None
        self.last_terminal_excitation_: np.ndarray | None = None
        self.fiedler_value_: float = 0.0
        self.mixing_time_: float = float("inf")
        self.fiedler_vector_: np.ndarray | None = None
        self.hawkes_branching_matrix_: np.ndarray | None = None
        self.hawkes_spectral_radius_: float = 0.0
        self.last_diagnostics_: SpectralHawkesDiagnostics | None = None

    @staticmethod
    def build_precision_graph_laplacian(
        returns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the Ledoit-Wolf shrunk correlation matrix, precision matrix,
        non-negative partial correlation adjacency W, and normalized Graph Laplacian L.
        """
        n_obs, n_assets = returns.shape
        if n_assets == 1:
            return (
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
                np.ones((1, 1), dtype=np.float64),
            )

        std = np.std(returns, axis=0, ddof=1)
        std = np.where(std > 1e-10, std, 1.0)
        z_ret = (returns - np.mean(returns, axis=0)) / std

        if n_obs >= 3:
            try:
                lw = LedoitWolf().fit(z_ret)
                corr = lw.covariance_.astype(np.float64)
            except Exception:
                corr = np.cov(z_ret, rowvar=False)
        else:
            corr = np.eye(n_assets, dtype=np.float64)

        d_inv = 1.0 / np.sqrt(np.clip(np.diag(corr), 1e-10, None))
        corr = corr * np.outer(d_inv, d_inv)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        theta = np.linalg.inv(corr + 1e-3 * np.eye(n_assets, dtype=np.float64))
        theta_diag = np.sqrt(np.clip(np.diag(theta), 1e-10, None))
        partial_corr = -theta / np.outer(theta_diag, theta_diag)

        adjacency = np.maximum(0.0, partial_corr)
        np.fill_diagonal(adjacency, 0.0)

        deg = np.sum(adjacency, axis=1)
        deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
        norm_adj = adjacency * np.outer(deg_inv_sqrt, deg_inv_sqrt)
        laplacian = np.eye(n_assets, dtype=np.float64) - norm_adj
        laplacian = 0.5 * (laplacian + laplacian.T)

        return laplacian, norm_adj, corr

    def _rhs(
        self,
        v: np.ndarray,
        u: np.ndarray,
        laplacian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the coupled FitzHugh-Nagumo Graph Reaction-Diffusion vector field.
        """
        v_clipped = np.clip(v, -2.0, 2.5)
        diffusion_v = -self.diffusion_coeff * (laplacian @ v_clipped)
        cubic_reaction = (
            self.reaction_gain
            * v_clipped
            * (v_clipped - self.critical_threshold)
            * (1.0 - v_clipped)
        )
        dv = (
            diffusion_v
            - self.mean_reversion * v_clipped
            + cubic_reaction
            - self.inhibitor_coupling * u
        )

        diffusion_u = -self.inhibitor_diffusion * (laplacian @ u)
        du = diffusion_u + self.recovery_rate * (v_clipped - 0.8 * u)
        return dv, du

    def integrate_reaction_diffusion(
        self,
        v0: np.ndarray,
        u0: np.ndarray,
        laplacian: np.ndarray,
        horizon_steps: int,
        dt: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrates the Graph Reaction-Diffusion ODE system via 4th-order Runge-Kutta (RK4).
        """
        v = v0.astype(np.float64, copy=True)
        u = u0.astype(np.float64, copy=True)
        n_substeps = max(1, int(round(horizon_steps / dt)))
        h = float(horizon_steps) / float(n_substeps)

        for _ in range(n_substeps):
            k1_v, k1_u = self._rhs(v, u, laplacian)
            k2_v, k2_u = self._rhs(v + 0.5 * h * k1_v, u + 0.5 * h * k1_u, laplacian)
            k3_v, k3_u = self._rhs(v + 0.5 * h * k2_v, u + 0.5 * h * k2_u, laplacian)
            k4_v, k4_u = self._rhs(v + h * k3_v, u + h * k3_u, laplacian)

            v = v + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            u = u + (h / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
            v = np.clip(v, -1.5, 2.5)
            u = np.clip(u, -1.5, 2.5)

        return v, u

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.DataFrame:
        tickers = list(data.prices.columns)
        returns_df = data.prices.pct_change().dropna(how="all").fillna(0.0)

        slow_slice = returns_df.iloc[-self.slow_window :]
        fast_slice = returns_df.iloc[-self.fast_window :]

        laplacian, norm_adj, corr_base = self.build_precision_graph_laplacian(
            slow_slice.to_numpy(dtype=np.float64)
        )
        self.last_laplacian_ = laplacian
        self.last_adjacency_ = norm_adj

        diag = compute_spectral_hawkes_diagnostics(
            returns=slow_slice.to_numpy(dtype=np.float64),
            laplacian=laplacian,
        )
        self.fiedler_value_ = diag.fiedler_value
        self.mixing_time_ = diag.mixing_time
        self.fiedler_vector_ = diag.fiedler_vector
        self.hawkes_branching_matrix_ = diag.hawkes_branching_matrix
        self.hawkes_spectral_radius_ = diag.hawkes_spectral_radius
        self.last_diagnostics_ = diag

        sigma_slow = slow_slice.std(ddof=1).fillna(0.01).to_numpy(
            dtype=np.float64
        ) * np.sqrt(self.annualization_factor)
        sigma_slow = np.clip(sigma_slow, 1e-4, 5.0)

        sigma_fast = fast_slice.std(ddof=1).fillna(0.01).to_numpy(
            dtype=np.float64
        ) * np.sqrt(self.annualization_factor)
        sigma_fast = np.clip(sigma_fast, 1e-4, 5.0)

        if self.implied_vol_feature in data.features:
            iv_latest = (
                data.features[self.implied_vol_feature]
                .iloc[-1]
                .reindex(tickers)
                .to_numpy(dtype=np.float64)
            )
            valid = np.isfinite(iv_latest) & (iv_latest > 0.0)
            sigma_init = np.where(valid, 0.7 * iv_latest + 0.3 * sigma_fast, sigma_fast)
        else:
            sigma_init = sigma_fast

        v0 = np.log(np.clip(sigma_init, 1e-4, 5.0) / sigma_slow)
        # Augment initial excitation with causal Wiener-Hopf Hawkes jump spillover
        if self.hawkes_branching_matrix_ is not None:
            v0 = v0 + 0.15 * (self.hawkes_branching_matrix_ @ np.maximum(0.0, v0))
        u0 = 0.25 * np.maximum(0.0, v0)

        v_term, _ = self.integrate_reaction_diffusion(
            v0=v0,
            u0=u0,
            laplacian=laplacian,
            horizon_steps=self.horizon_steps,
        )
        self.last_terminal_excitation_ = v_term

        sigma_forecast = sigma_slow * np.exp(np.clip(v_term, -1.5, 2.5))

        pos_excitation = np.maximum(0.0, v_term)
        mean_excitation = float(np.mean(pos_excitation))
        contagion_weight = float(np.tanh(mean_excitation) * 0.35)

        graph_corr = 0.5 * (corr_base + norm_adj)
        np.fill_diagonal(graph_corr, 1.0)
        eigvals, eigvecs = np.linalg.eigh(graph_corr)
        eigvals = np.maximum(eigvals, 1e-4)
        graph_corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_inv = 1.0 / np.sqrt(np.clip(np.diag(graph_corr_psd), 1e-10, None))
        graph_corr_psd = graph_corr_psd * np.outer(d_inv, d_inv)
        np.fill_diagonal(graph_corr_psd, 1.0)

        corr_forecast = (
            1.0 - contagion_weight
        ) * corr_base + contagion_weight * graph_corr_psd
        np.fill_diagonal(corr_forecast, 1.0)

        cov_forecast = np.outer(sigma_forecast, sigma_forecast) * corr_forecast
        cov_forecast = 0.5 * (cov_forecast + cov_forecast.T)

        return pd.DataFrame(cov_forecast, index=tickers, columns=tickers)
