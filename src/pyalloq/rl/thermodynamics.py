from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as sla


@dataclass
class ThermodynamicDiagnostics:
    """
    Diagnostic metrics from the Thermodynamic & Bures-Wasserstein Regime Gate
    (Moroke et al. 2026, Research Section 8.2).
    """

    wasserstein_distance: float
    temp_fast: float
    temp_slow: float
    carnot_efficiency: float
    damping_factor: float
    should_halt_rebalance: bool


class WassersteinRegimeGate:
    """
    Thermodynamic & Optimal-Transport Regime Gate (Moroke et al. 2026, Research Section 8.2).

    Models the portfolio rebalancing loop as a non-equilibrium stochastic heat engine
    operating between a fast (recent window) market bath at temperature T_fast = Tr(Sigma_fast)/N
    and a slow (structural window) bath at temperature T_slow = Tr(Sigma_slow)/N.

    1. Bures-Wasserstein 2-Distance between Gaussian regimes N(mu_1, Sigma_1) and N(mu_2, Sigma_2):
       W_2^2 = ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2 (Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2})
    2. Carnot Efficiency Bound on extractable alpha per unit of entropy dissipation:
       eta_Carnot = 1 - min(T_fast, T_slow) / max(T_fast, T_slow)
    3. Rebalance Gating & Damping:
       When W_2 exceeds `w2_threshold` (structural regime rupture / frictional dissipation spike),
       dampens or halts rebalancing trades toward pre-trade weights w_{t^-}.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 60,
        w2_threshold: float = 0.03,
        sensitivity: float = 25.0,
        min_damping: float = 0.0,
    ) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.w2_threshold = w2_threshold
        self.sensitivity = sensitivity
        self.min_damping = min_damping

    @staticmethod
    def bures_wasserstein_distance(
        mu1: np.ndarray,
        cov1: np.ndarray,
        mu2: np.ndarray,
        cov2: np.ndarray,
    ) -> float:
        """
        Compute exact Bures-Wasserstein W_2 distance between N(mu1, cov1) and N(mu2, cov2).
        """
        n = len(mu1)
        eps_eye = np.eye(n, dtype=float) * 1e-8
        c1 = 0.5 * (cov1 + cov1.T) + eps_eye
        c2 = 0.5 * (cov2 + cov2.T) + eps_eye

        mean_diff_sq = float(np.sum((mu1 - mu2) ** 2))

        sqrt_c1 = np.real(sla.sqrtm(c1))
        inner = sqrt_c1 @ c2 @ sqrt_c1
        inner = 0.5 * (inner + inner.T) + eps_eye
        sqrt_inner = np.real(sla.sqrtm(inner))

        bures_sq = float(np.trace(c1 + c2 - 2.0 * sqrt_inner))
        w2_sq = max(0.0, mean_diff_sq + bures_sq)
        return float(np.sqrt(w2_sq))

    def evaluate(self, returns: pd.DataFrame) -> ThermodynamicDiagnostics:
        """
        Evaluate thermodynamic regime metrics from a DataFrame of asset returns.
        """
        clean_ret = returns.dropna(how="all").fillna(0.0)
        n_obs, n_assets = clean_ret.shape
        if n_obs < max(5, self.fast_window):
            return ThermodynamicDiagnostics(
                wasserstein_distance=0.0,
                temp_fast=1e-4,
                temp_slow=1e-4,
                carnot_efficiency=0.0,
                damping_factor=1.0,
                should_halt_rebalance=False,
            )

        fast_slice = clean_ret.iloc[-self.fast_window :]
        slow_slice = clean_ret.iloc[-min(n_obs, self.slow_window) :]

        mu_fast = fast_slice.mean().to_numpy(dtype=float)
        cov_fast = fast_slice.cov().fillna(0.0).to_numpy(dtype=float)
        mu_slow = slow_slice.mean().to_numpy(dtype=float)
        cov_slow = slow_slice.cov().fillna(0.0).to_numpy(dtype=float)

        w2 = self.bures_wasserstein_distance(mu_fast, cov_fast, mu_slow, cov_slow)

        temp_fast = max(float(np.trace(cov_fast)) / max(1, n_assets), 1e-10)
        temp_slow = max(float(np.trace(cov_slow)) / max(1, n_assets), 1e-10)
        t_min = min(temp_fast, temp_slow)
        t_max = max(temp_fast, temp_slow)
        carnot_eff = float(np.clip(1.0 - (t_min / t_max), 0.0, 1.0))

        # Smooth exponential damping when W2 exceeds threshold
        excess_w2 = max(0.0, w2 - self.w2_threshold)
        damping = float(
            np.clip(
                np.exp(-self.sensitivity * excess_w2),
                self.min_damping,
                1.0,
            )
        )
        should_halt = bool(w2 > 2.0 * self.w2_threshold and damping <= 0.1)

        return ThermodynamicDiagnostics(
            wasserstein_distance=w2,
            temp_fast=temp_fast,
            temp_slow=temp_slow,
            carnot_efficiency=carnot_eff,
            damping_factor=damping,
            should_halt_rebalance=should_halt,
        )

    def gate_weights(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series | None,
        returns: pd.DataFrame,
    ) -> tuple[pd.Series, ThermodynamicDiagnostics]:
        """
        Blend `target_weights` and `current_weights` using the thermodynamic damping factor:
            w_gated = damping * w_target + (1 - damping) * w_current
        """
        diag = self.evaluate(returns)
        if current_weights is None:
            return target_weights, diag

        w_curr = current_weights.reindex(target_weights.index).fillna(0.0)
        c_sum = float(w_curr.sum())
        if c_sum <= 0:
            return target_weights, diag
        w_curr = w_curr / c_sum

        lam = 0.0 if diag.should_halt_rebalance else diag.damping_factor
        gated = lam * target_weights + (1.0 - lam) * w_curr
        g_sum = float(gated.sum())
        if g_sum > 0:
            gated = gated / g_sum
        return gated, diag
