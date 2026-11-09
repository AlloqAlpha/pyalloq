from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralHawkesDiagnostics:
    """
    Diagnostic metrics from Spectral Graph Laplacian & Wiener-Hopf Multivariate Hawkes analysis.
    """

    fiedler_value: float
    mixing_time: float
    fiedler_vector: np.ndarray
    hawkes_branching_matrix: np.ndarray
    hawkes_spectral_radius: float
    fragility_shrinkage_alpha: float


class SpectralGraphFragility:
    """
    Spectral Filtering and Algebraic Connectivity Analyzer for Financial Graph Laplacians.

    Extracts:
    - The Fiedler Value (lambda_2, second smallest eigenvalue of the Graph Laplacian L),
      measuring algebraic connectivity and systemic cascade propensity.
    - The Network Mixing Time (tau_mix = 1 / max(lambda_2, eps)), measuring the speed at
      which localized liquidity/volatility shocks equilibrate across the asset graph.
    - The Fiedler Vector (v_2 in R^N), providing the optimal spectral bisection coordinate
      for topological portfolio cuts (used by `SpectralLaplacianHRPAllocator`).
    """

    @staticmethod
    def analyze_laplacian(
        laplacian: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """
        Returns `(fiedler_value, mixing_time, fiedler_vector)` for a symmetric Graph Laplacian `L`.
        """
        n = laplacian.shape[0]
        if n <= 1:
            return 0.0, float("inf"), np.ones(n, dtype=np.float64)

        sym_L = 0.5 * (laplacian + laplacian.T)
        eigvals, eigvecs = np.linalg.eigh(sym_L)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        fiedler_val = float(max(0.0, eigvals[1]))
        mixing_time = float(1.0 / max(fiedler_val, 1e-6))
        fiedler_vec = eigvecs[:, 1].astype(np.float64)
        # Canonical sign orientation using largest-magnitude coordinate for deterministic spectral cuts
        idx_max = int(np.argmax(np.abs(fiedler_vec)))
        if fiedler_vec[idx_max] < 0:
            fiedler_vec = -fiedler_vec

        return fiedler_val, mixing_time, fiedler_vec


class WienerHopfHawkesSolver:
    """
    Non-Parametric Wiener-Hopf Multivariate Hawkes Process (MHP) Solver (Bacry et al. 2016).

    Models the D-dimensional self-exciting point process of asset volatility/return jumps:
        lambda_i(t) = mu_i + sum_{j=1}^D int_0^t phi_{ij}(t - s) dN_j(s)

    Solves the discretized Wiener-Hopf integral system over Q quadrature lag nodes {tau_q, w_q}
    relating empirical jump conditional covariance density C(tau) to the causal excitation
    kernels phi_{ij}(tau):
        C_{ij}(tau) = Lambda_j phi_{ij}(tau) + sum_{k=1}^D int_0^infty phi_{ik}(u) C_{kj}(tau - u) du

    Extracts:
    - `branching_matrix` G in R_+^{D x D} where G_{ij} = ||phi_{ij}||_1 = int_0^infty phi_{ij}(s) ds
      is the expected number of direct offspring jumps on asset i triggered by a jump on asset j.
    - `spectral_radius` rho(G) = max_k |lambda_k(G)| in [0, 1), which serves as a real-time
      early warning indicator for critical systemic liquidity cascades as rho(G) -> 1.
    """

    def __init__(
        self,
        max_lag: int = 5,
        jump_z_threshold: float = 1.75,
        ridge_reg: float = 1e-2,
        max_spectral_radius: float = 0.98,
    ) -> None:
        self.max_lag = int(max(1, max_lag))
        self.jump_z_threshold = float(jump_z_threshold)
        self.ridge_reg = float(ridge_reg)
        self.max_spectral_radius = float(max_spectral_radius)

    def fit_from_returns(
        self,
        returns: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Extracts jump point-process increments dN_t in {0, 1}^{T x N} from asset returns,
        solves the Nyström Wiener-Hopf system for the excitation kernels Phi(tau), and returns
        `(branching_matrix, spectral_radius, baseline_intensity)`.
        """
        n_obs, n_assets = returns.shape
        if n_obs < self.max_lag + 3:
            zeros = np.zeros((n_assets, n_assets), dtype=np.float64)
            return zeros, 0.0, np.full(n_assets, 0.05, dtype=np.float64)

        std = np.std(returns, axis=0, ddof=1)
        std = np.where(std > 1e-10, std, 1.0)
        z_scores = np.abs(returns - np.mean(returns, axis=0)) / std

        # Discrete jump indicator process dN_{t, i} in {0, 1}
        dN = (z_scores >= self.jump_z_threshold).astype(np.float64)
        mean_rate = np.clip(np.mean(dN, axis=0), 1e-4, 0.95)
        centered_dN = dN - mean_rate[None, :]

        Q = min(self.max_lag, max(1, n_obs // 4))
        # Empirical lag covariance matrices C(h) = E[(dN_{t} - Lambda)(dN_{t-h} - Lambda)^T]
        # C_lags[h - 1] has shape (N, N) for h = 1, ..., Q, plus C_0 at lag 0
        C_0 = (centered_dN.T @ centered_dN) / float(n_obs) + self.ridge_reg * np.eye(
            n_assets, dtype=np.float64
        )
        C_lags = np.zeros((Q, n_assets, n_assets), dtype=np.float64)
        for h in range(1, Q + 1):
            C_lags[h - 1] = (centered_dN[h:].T @ centered_dN[:-h]) / float(n_obs - h)

        # Assemble Nyström block-Toeplitz Wiener-Hopf operator K_WH of shape (Q * N, Q * N)
        # where block (q, r) is C(|q - r|)^T
        block_size = Q * n_assets
        K_wh = np.zeros((block_size, block_size), dtype=np.float64)
        rhs = np.zeros((block_size, n_assets), dtype=np.float64)

        for q in range(Q):
            q_slice = slice(q * n_assets, (q + 1) * n_assets)
            rhs[q_slice, :] = C_lags[q].T
            for r in range(Q):
                r_slice = slice(r * n_assets, (r + 1) * n_assets)
                diff = q - r
                if diff == 0:
                    block = C_0
                elif diff > 0:
                    block = C_lags[diff - 1]
                else:
                    block = C_lags[-diff - 1].T
                K_wh[q_slice, r_slice] = block

        K_wh += self.ridge_reg * np.eye(block_size, dtype=np.float64)
        # Solve K_wh @ Phi_stack = rhs -> Phi_stack[q] = Phi(tau_q)^T
        try:
            phi_stack = np.linalg.solve(K_wh, rhs)
        except np.linalg.LinAlgError:
            phi_stack = np.linalg.lstsq(K_wh, rhs, rcond=None)[0]

        # Project onto non-negative causal excitation kernels and integrate L1 norm over lags
        kernels = np.zeros((Q, n_assets, n_assets), dtype=np.float64)
        for q in range(Q):
            q_slice = slice(q * n_assets, (q + 1) * n_assets)
            kernels[q] = np.maximum(0.0, phi_stack[q_slice, :].T)

        # Quadrature integration G_{ij} = sum_{q=1}^Q w_q phi_{ij}(q)
        # Normalize lag weights so G represents expected offspring per jump event
        branching = np.sum(kernels, axis=0) / float(max(1, Q))

        eigvals = np.linalg.eigvals(branching)
        raw_radius = float(np.max(np.abs(eigvals))) if len(eigvals) > 0 else 0.0
        if raw_radius > self.max_spectral_radius and raw_radius > 1e-12:
            scale = self.max_spectral_radius / raw_radius
            branching = branching * scale
            spec_radius = self.max_spectral_radius
        else:
            spec_radius = raw_radius

        baseline_mu = mean_rate * max(0.05, 1.0 - spec_radius)
        return branching, float(spec_radius), baseline_mu


def compute_spectral_hawkes_diagnostics(
    returns: np.ndarray,
    laplacian: np.ndarray,
    max_lag: int = 5,
) -> SpectralHawkesDiagnostics:
    """
    Convenience helper computing both Graph Laplacian spectral fragility (Fiedler value lambda_2,
    mixing time, Fiedler vector v_2) and Wiener-Hopf Multivariate Hawkes branching metrics.
    """
    fiedler_val, mix_time, fiedler_vec = SpectralGraphFragility.analyze_laplacian(
        laplacian
    )
    wh_solver = WienerHopfHawkesSolver(max_lag=max_lag)
    branching_mat, spec_radius, _ = wh_solver.fit_from_returns(returns)

    # Combine Fiedler algebraic connectivity and Hawkes criticality into shrinkage alpha in [0.05, 0.90]
    criticality = spec_radius / max(1.0 - min(spec_radius, 0.95), 0.05)
    alpha = float(
        np.clip(
            0.10
            + 0.25 * np.tanh(0.5 * fiedler_val)
            + 0.35 * np.tanh(0.5 * criticality),
            0.05,
            0.90,
        )
    )
    return SpectralHawkesDiagnostics(
        fiedler_value=fiedler_val,
        mixing_time=mix_time,
        fiedler_vector=fiedler_vec,
        hawkes_branching_matrix=branching_mat,
        hawkes_spectral_radius=spec_radius,
        fragility_shrinkage_alpha=alpha,
    )
