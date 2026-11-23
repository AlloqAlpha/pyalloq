from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import MarketData
from .interfaces import BaseReturnEstimator, BaseViewGenerator

ConformalMethod = Literal["pid", "dtaci", "nexcp"]


@dataclass
class ConformalIntervalResult:
    """
    Holds distribution-free calibrated prediction intervals and Black-Litterman
    uncertainty matrix produced by `ConformalPIDCalibrator`.
    """

    point_forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    interval_width: pd.Series
    calibrated_alpha: float
    empirical_coverage: float
    omega_matrix: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class ConformalPIDCalibrator(BaseReturnEstimator, BaseViewGenerator):
    """
    Time-Series Online Conformal Calibrator (`ConformalPIDCalibrator`).

    Wraps any `BaseReturnEstimator` (or operates directly on online residual score streams)
    to emit distribution-free, finite-sample prediction intervals under non-exchangeable
    financial time-series shifts using one of three peer-reviewed frameworks:

    1. **`method="pid"` (Conformal PID Control, Angelopoulos, Candès & Tibshirani, *NeurIPS* 2023)**:
       - **P control (Quantile tracking)**: Online gradient descent on pinball loss
         `q_{t+1} = q_t + eta * (err_t - alpha)`.
       - **I control (Error integration)**: Nonlinear saturating integrator `r_t(sum_{tau <= t} (err_tau - alpha))`
         using scaled `tan / arctan` saturation guaranteeing exact long-run coverage `1 - alpha`
         even under unbounded fat-tailed return shocks.
       - **D control (Scorecasting)**: Lightweight EWMA volatility scorecaster predicting the
         next residual spike prospectively so interval widths expand *during* volatility spikes
         rather than reacting late.
    2. **`method="dtaci"` (Dynamically Tuned Adaptive Conformal Inference, Gibbs & Candès,
       *NeurIPS* 2021 / *JMLR* 2024)**:
       Adapts miscoverage `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)` across a grid
       of candidate step sizes `gamma in gamma_grid` aggregated via exponentially weighted experts.
    3. **`method="nexcp"` (Non-Exchangeable Conformal Prediction, Barber, Candès, Ramdas &
       Tibshirani, *Annals of Statistics* 2023)**:
       Replaces uniform split-conformal weights with exponentially decayed weights `w_i = rho^{t-i}`
       so recent regime residuals govern interval width.

    Also implements `BaseReturnEstimator` and `BaseViewGenerator` (`.generate(data) -> (P, Q, Omega)`)
    so it can wrap `TimesFM`, `TFTDeepReturnsNet`, or `EnsembleReturnEstimator` directly inside
    `BlackLittermanEstimator` or `ConformalKellyOverlay`.
    """

    def __init__(
        self,
        base_estimator: BaseReturnEstimator | None = None,
        alpha: float = 0.25,
        method: ConformalMethod = "pid",
        learning_rate_p: float = 0.15,
        integral_gain_i: float = 0.08,
        derivative_gain_d: float = 0.35,
        nexcp_decay: float = 0.94,
        dtaci_gamma_grid: Sequence[float] = (0.005, 0.02, 0.05, 0.10, 0.20),
        calibration_steps: int = 30,
        annualization_factor: float = 252.0,
    ) -> None:
        if not (0.01 <= alpha <= 0.95):
            raise ValueError("`alpha` miscoverage rate must be in [0.01, 0.95].")
        if method not in ("pid", "dtaci", "nexcp"):
            raise ValueError("`method` must be one of 'pid', 'dtaci', or 'nexcp'.")

        self.base_estimator = base_estimator
        self.alpha = float(alpha)
        self.method: ConformalMethod = method
        self.learning_rate_p = float(max(1e-4, learning_rate_p))
        self.integral_gain_i = float(max(0.0, integral_gain_i))
        self.derivative_gain_d = float(max(0.0, derivative_gain_d))
        self.nexcp_decay = float(np.clip(nexcp_decay, 0.50, 0.999))
        self.dtaci_gamma_grid = tuple(float(g) for g in dtaci_gamma_grid)
        self.calibration_steps = max(5, int(calibration_steps))
        self.annualization_factor = float(annualization_factor)

        self.last_interval_result_: ConformalIntervalResult | None = None
        self.last_posterior_std_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}

    @staticmethod
    def _saturating_tan_integrator(
        cum_error: np.ndarray,
        t_len: int,
        c_sat: float = 1.5,
    ) -> np.ndarray:
        """
        Angelopoulos, Candès & Tibshirani (NeurIPS 2023) saturating tan-like integrator
        mapping cumulative coverage error `sum (err_tau - alpha)` into bounded quantile correction.
        """
        scale = max(math.sqrt(max(1, t_len) * math.log(max(3, t_len))), 1.0)
        normalized = np.clip(cum_error / (c_sat * scale), -1.35, 1.35)
        return np.tan(normalized)

    def calibrate_residual_series(
        self,
        residual_scores: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """
        Runs online conformal quantile calibration over a `(T_cal, N_assets)` matrix of
        non-negative conformity scores `s_t = |y_t - y_hat_t|`.

        Returns
        -------
        tuple[np.ndarray, float, float]
            `(next_calibrated_radius_per_asset, effective_alpha, realized_empirical_coverage)`
        """
        if residual_scores.ndim == 1:
            residual_scores = residual_scores[:, None]
        t_len, n_assets = residual_scores.shape
        if t_len == 0:
            return (
                np.full(n_assets, 0.05, dtype=np.float64),
                self.alpha,
                1.0 - self.alpha,
            )

        s_mat = np.maximum(np.asarray(residual_scores, dtype=np.float64), 1e-8)
        score_scale = np.maximum(np.median(s_mat, axis=0), 1e-5)

        if self.method == "nexcp":
            # Barber et al. (2023) exponentially decayed weighted quantile + coverage tracking
            covered_list: list[float] = []
            q_curr = np.quantile(s_mat[: max(2, t_len // 4)], 1.0 - self.alpha, axis=0)
            for t in range(1, t_len):
                err_t = (s_mat[t] > q_curr).astype(np.float64)
                covered_list.append(1.0 - float(np.mean(err_t)))
                hist_s = s_mat[: t + 1]
                weights = np.array(
                    [self.nexcp_decay ** (t - i) for i in range(t + 1)],
                    dtype=np.float64,
                )
                weights /= np.sum(weights)
                for j in range(n_assets):
                    sorter = np.argsort(hist_s[:, j])
                    s_sorted = hist_s[sorter, j]
                    w_sorted = np.cumsum(weights[sorter])
                    idx_q = int(np.searchsorted(w_sorted, 1.0 - self.alpha))
                    idx_q = min(idx_q, len(s_sorted) - 1)
                    q_curr[j] = float(s_sorted[idx_q])
            realized_cov = (
                float(np.mean(covered_list)) if covered_list else (1.0 - self.alpha)
            )
            return np.maximum(q_curr, 1e-6), self.alpha, realized_cov

        if self.method == "dtaci":
            # Gibbs & Candès (2021, 2024) DtACI across candidate gamma step sizes
            n_gammas = len(self.dtaci_gamma_grid)
            alphas_m = np.full((n_gammas, n_assets), self.alpha, dtype=np.float64)
            expert_weights = np.full(
                (n_gammas, n_assets), 1.0 / n_gammas, dtype=np.float64
            )
            q_curr = np.quantile(s_mat[: max(2, t_len // 4)], 1.0 - self.alpha, axis=0)
            covered_list = []
            eff_alpha_curr = np.full(n_assets, self.alpha, dtype=np.float64)

            for t in range(1, t_len):
                err_agg = (s_mat[t] > q_curr).astype(np.float64)
                covered_list.append(1.0 - float(np.mean(err_agg)))
                hist_s = s_mat[: t + 1]

                q_candidates = np.zeros((n_gammas, n_assets), dtype=np.float64)
                for m_idx, gam in enumerate(self.dtaci_gamma_grid):
                    lvl = np.clip(1.0 - alphas_m[m_idx], 0.02, 0.995)
                    for j in range(n_assets):
                        q_candidates[m_idx, j] = float(
                            np.quantile(hist_s[:, j], lvl[j])
                        )
                    err_m = (s_mat[t] > q_candidates[m_idx]).astype(np.float64)
                    # Pinball loss for expert aggregation
                    diff_m = s_mat[t] - q_candidates[m_idx]
                    pinball_m = (
                        np.where(
                            diff_m >= 0.0,
                            (1.0 - alphas_m[m_idx]) * diff_m,
                            -alphas_m[m_idx] * diff_m,
                        )
                        / score_scale
                    )
                    expert_weights[m_idx] *= np.exp(
                        -0.5 * np.clip(pinball_m, 0.0, 10.0)
                    )
                    # ACI update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
                    alphas_m[m_idx] = np.clip(
                        alphas_m[m_idx] + gam * (self.alpha - err_m), 0.005, 0.95
                    )

                # Fixed-share expert mixing for non-stationarity
                expert_weights /= np.sum(expert_weights, axis=0, keepdims=True)
                expert_weights = 0.95 * expert_weights + 0.05 / n_gammas
                q_curr = np.sum(expert_weights * q_candidates, axis=0)
                eff_alpha_curr = np.sum(expert_weights * alphas_m, axis=0)

            realized_cov = (
                float(np.mean(covered_list)) if covered_list else (1.0 - self.alpha)
            )
            return (
                np.maximum(q_curr, 1e-6),
                float(np.mean(eff_alpha_curr)),
                realized_cov,
            )

        # Default: method == "pid" (Conformal PID Control, Angelopoulos, Candès & Tibshirani, NeurIPS 2023)
        q_p = np.quantile(s_mat[: max(2, t_len // 4)], 1.0 - self.alpha, axis=0)
        cum_err = np.zeros(n_assets, dtype=np.float64)
        ewma_score = q_p.copy()
        q_curr = q_p.copy()
        covered_list = []

        for t in range(t_len):
            s_t = s_mat[t]
            err_t = (s_t > q_curr).astype(np.float64)
            if t >= 1:
                covered_list.append(1.0 - float(np.mean(err_t)))

            # 1. P-control: online quantile tracking via pinball subgradient (err_t - alpha)
            q_p = np.maximum(
                q_p + self.learning_rate_p * score_scale * (err_t - self.alpha),
                1e-6,
            )
            # 2. I-control: saturating tan error integration
            cum_err += err_t - self.alpha
            i_term = (
                self.integral_gain_i
                * score_scale
                * self._saturating_tan_integrator(cum_err, t + 1)
            )
            # 3. D-control (Scorecaster): prospective EWMA volatility / residual spike forecast
            ewma_score = 0.30 * s_t + 0.70 * ewma_score
            d_term = self.derivative_gain_d * (ewma_score - q_p)

            q_curr = np.maximum(q_p + i_term + d_term, 1e-6)

        realized_cov = (
            float(np.mean(covered_list)) if covered_list else (1.0 - self.alpha)
        )
        return q_curr, self.alpha, realized_cov

    def calibrate(self, data: MarketData) -> ConformalIntervalResult:
        """
        Computes the point return forecast and online conformal prediction interval
        `[lower_bound, upper_bound]` for each asset in `data`.
        """
        assets = list(data.prices.columns)
        returns_df = data.prices.pct_change().dropna()
        ret_np = returns_df.to_numpy(dtype=np.float64)

        if self.base_estimator is not None:
            point_ser = self.base_estimator.estimate(data).reindex(assets).fillna(0.0)
        else:
            ann_mu = (
                returns_df.mean().reindex(assets).fillna(0.0)
                * self.annualization_factor
                if len(returns_df) > 0
                else pd.Series(0.0, index=assets)
            )
            point_ser = ann_mu

        if len(ret_np) < 4:
            half_w = pd.Series(0.10, index=assets)
            omega_df = pd.DataFrame(
                np.diag(half_w.to_numpy() ** 2), index=assets, columns=assets
            )
            res = ConformalIntervalResult(
                point_forecast=point_ser,
                lower_bound=point_ser - half_w,
                upper_bound=point_ser + half_w,
                interval_width=2.0 * half_w,
                calibrated_alpha=self.alpha,
                empirical_coverage=1.0 - self.alpha,
                omega_matrix=omega_df,
                metadata={"method": self.method, "target_coverage": 1.0 - self.alpha},
            )
            self.last_interval_result_ = res
            return res

        # Build rolling causal residual conformity scores s_t = |r_t * ann_factor - mu_hat_{t-1}|
        t_cal = min(len(ret_np) - 2, self.calibration_steps)
        ann_rets = ret_np * self.annualization_factor
        # Fast causal EWMA / base forecast residual stream
        ewma_preds = (
            returns_df.ewm(span=15, min_periods=1)
            .mean()
            .shift(1)
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            * self.annualization_factor
        )
        residuals = np.abs(ann_rets[-t_cal:] - ewma_preds[-t_cal:])

        radii_ann, eff_alpha, emp_cov = self.calibrate_residual_series(residuals)
        half_width_ser = pd.Series(radii_ann, index=assets, name="conformal_radius")
        lower_ser = (point_ser - half_width_ser).rename("lower_bound")
        upper_ser = (point_ser + half_width_ser).rename("upper_bound")
        width_ser = (2.0 * half_width_ser).rename("interval_width")

        # Convert calibrated interval width to equivalent Gaussian standard deviation for BL Omega
        z_equiv = max(float(norm.ppf(1.0 - 0.5 * self.alpha)), 0.5)
        equiv_std = np.maximum(radii_ann / z_equiv, 1e-4)
        self.last_posterior_std_ = pd.Series(
            equiv_std, index=assets, name="posterior_std"
        )
        self.last_quantiles_ = {
            0.5 * self.alpha: lower_ser,
            0.50: point_ser,
            1.0 - 0.5 * self.alpha: upper_ser,
        }
        omega_df = pd.DataFrame(np.diag(equiv_std**2), index=assets, columns=assets)

        res = ConformalIntervalResult(
            point_forecast=point_ser,
            lower_bound=lower_ser,
            upper_bound=upper_ser,
            interval_width=width_ser,
            calibrated_alpha=eff_alpha,
            empirical_coverage=emp_cov,
            omega_matrix=omega_df,
            metadata={
                "method": self.method,
                "target_coverage": 1.0 - self.alpha,
                "empirical_coverage": emp_cov,
                "effective_alpha": eff_alpha,
            },
        )
        self.last_interval_result_ = res
        return res

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        res = self.calibrate(data)
        return res.point_forecast

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        res = self.calibrate(data)
        assets = list(data.prices.columns)
        n_assets = len(assets)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        return P, res.point_forecast, res.omega_matrix


class ConformalRiskController:
    """
    Conformal Risk Control (`ConformalRiskController`, Angelopoulos, Bates, Fisch,
    Lei & Schuster, *ICLR* 2024, arXiv:2208.02814).

    Generalizes binary conformal miscoverage to any bounded monotone non-increasing risk loss
    `L_i(lambda) in [0, B]` (such as tail-loss exceedance, drawdown breach magnitude, or
    crash false-negative rate), selecting the smallest safety parameter `lambda_hat`:

        lambda_hat = inf { lambda in Lambda : (n / (n + 1)) * R_hat_n(lambda) + B / (n + 1) <= alpha }

    which guarantees finite-sample risk control `E[L_{n+1}(lambda_hat)] <= alpha` tight up to `O(1/n)`.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        bound_b: float = 1.0,
        lambda_grid: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if not (0.0 < alpha <= bound_b):
            raise ValueError("`alpha` risk budget must be in (0, bound_b].")
        self.alpha = float(alpha)
        self.bound_b = float(max(1e-4, bound_b))
        self.lambda_grid = (
            np.asarray(lambda_grid, dtype=np.float64)
            if lambda_grid is not None
            else np.linspace(0.0, 1.0, 101, dtype=np.float64)
        )
        self.calibrated_lambda_: float = float(self.lambda_grid[-1])
        self.empirical_risk_at_lambda_: float = 0.0

    def calibrate_from_loss_matrix(self, loss_matrix: np.ndarray) -> float:
        """
        Calibrates `lambda_hat` from an `(n_samples, len(lambda_grid))` matrix where entry
        `(i, m)` is `L_i(lambda_m) in [0, B]` (monotone non-increasing in `lambda_m`).
        """
        if loss_matrix.ndim == 1:
            loss_matrix = loss_matrix[:, None]
        n_samples, n_grid = loss_matrix.shape
        if n_samples == 0:
            self.calibrated_lambda_ = float(self.lambda_grid[-1])
            return self.calibrated_lambda_

        clipped_l = np.clip(
            np.asarray(loss_matrix, dtype=np.float64), 0.0, self.bound_b
        )
        r_hat = np.mean(clipped_l, axis=0)
        crc_objectives = (float(n_samples) / float(n_samples + 1)) * r_hat + (
            self.bound_b / float(n_samples + 1)
        )

        valid_indices = np.flatnonzero(crc_objectives <= self.alpha)
        if len(valid_indices) > 0:
            best_idx = int(valid_indices[0])
        else:
            best_idx = min(n_grid - 1, len(self.lambda_grid) - 1)

        self.calibrated_lambda_ = float(
            self.lambda_grid[min(best_idx, len(self.lambda_grid) - 1)]
        )
        self.empirical_risk_at_lambda_ = float(r_hat[min(best_idx, len(r_hat) - 1)])
        return self.calibrated_lambda_

    def calibrate(
        self,
        scores: np.ndarray,
        loss_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> float:
        """
        Calibrates `lambda_hat` over a 1D or 2D array of non-conformity / tail-loss scores.
        When `loss_fn=None`, uses the standard tail-exceedance loss
        `L_i(lambda) = min(B, max(0, (s_i - lambda) / scale))`.
        """
        s_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(s_arr) == 0:
            self.calibrated_lambda_ = float(self.lambda_grid[-1])
            return self.calibrated_lambda_

        s_scale = max(float(np.max(np.abs(s_arr))), 1e-6)
        grid = self.lambda_grid
        loss_mat = np.zeros((len(s_arr), len(grid)), dtype=np.float64)
        for m_idx, lam_val in enumerate(grid):
            if loss_fn is not None:
                loss_mat[:, m_idx] = loss_fn(s_arr, float(lam_val))
            else:
                thresh = float(lam_val) * s_scale
                loss_mat[:, m_idx] = np.clip(
                    np.maximum(0.0, s_arr - thresh) / s_scale,
                    0.0,
                    self.bound_b,
                )
        return self.calibrate_from_loss_matrix(loss_mat)
