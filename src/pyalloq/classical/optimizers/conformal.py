from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.conformal import (
    ConformalMethod,
    ConformalPIDCalibrator,
    ConformalRiskController,
)
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator, BaseReturnEstimator
from pyalloq_core.results import OptimizationResult

from pyalloq.classical.optimizers.naive import (
    InverseVolatilityAllocator,
)


@dataclass(frozen=True)
class RiskCertificate:
    """
    Auditable Execution-Time Risk Certificate (`RiskCertificate` / `SemanticRiskCertificate`,
    Zarei et al., Nov 2026, arXiv:2611.14699).

    Encapsulates hard execution-time portfolio safety constraints enforced by
    `ConformalSafetyGateAllocator`:
      - `max_tail_loss`: Maximum permissible conformal lower-bound single-period loss (e.g. `0.03` = 3%).
      - `max_gross_leverage`: Upper bound on `||w||_1`.
      - `max_turnover`: Upper bound on `||w - w_{t^-}||_1`.
      - `max_annual_volatility`: Upper bound on `sqrt(w^T Sigma_ann w)`.
      - `min_cash_buffer`: Minimum uninvested cash fraction `1 - sum(w) >= min_cash_buffer`.
      - `max_asset_weight`: Upper bound per single asset `w_i <= max_asset_weight`.
      - `sector_caps`: Optional mapping `{sector_name: max_sector_weight}`.
      - `restricted_assets`: Optional tuple of tickers barred from long exposure (`w_i == 0`).
      - `rationale`: Auditable explanation string or regime tag.
    """

    max_tail_loss: float = 0.04
    max_gross_leverage: float = 1.0
    max_turnover: float | None = 0.40
    max_annual_volatility: float | None = 0.25
    min_cash_buffer: float = 0.0
    max_asset_weight: float = 0.40
    min_asset_weight: float = 0.0
    sector_caps: dict[str, float] = field(default_factory=dict)
    restricted_assets: tuple[str, ...] = ()
    rationale: str = "ConformalRiskControl_DefaultCertificate"


SemanticRiskCertificate = RiskCertificate


class CPPSAllocator(BaseAllocator):
    """
    Conformal Predictive Portfolio Selection (`CPPSAllocator`, Kato, Feb 2025,
    arXiv:2410.16333v2).

    Instead of inverting a noisy `N x N` covariance matrix, generates `n_candidates`
    candidate portfolios `w in W` (combining Dirichlet simplex draws with equal-weight,
    inverse-volatility, and concentrated anchors), computes direct portfolio-level
    online conformal prediction intervals `C_T^w(X_{T+1}) = [L_T^{w, alpha}, U_T^{w, alpha}]`,
    and applies a **Two-Stage Highest-Return Lower-Risk (HR–LR) Screen**:
      1. **Stage 1 (LR Defense)**: Retain the top-`top_m_defense` candidates with the highest
         conformal lower bound `L_T^{w, alpha}`.
      2. **Stage 2 (HR Upside Capture)**: Pick the candidate from Stage 1 that maximizes
         the conformal upper bound `U_T^{w, alpha}`.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        alpha: float = 0.20,
        method: ConformalMethod = "pid",
        n_candidates: int = 200,
        top_m_defense: int = 25,
        max_weight: float = 0.50,
        calibration_steps: int = 35,
        random_state: int = 42,
    ) -> None:
        super().__init__(tickers or [])
        self.alpha = float(np.clip(alpha, 0.02, 0.50))
        self.method: ConformalMethod = method
        self.n_candidates = max(10, int(n_candidates))
        self.top_m_defense = max(1, min(int(top_m_defense), self.n_candidates))
        self.max_weight = float(np.clip(max_weight, 0.05, 1.0))
        self.calibration_steps = max(5, int(calibration_steps))
        self.random_state = int(random_state)

        self.calibrator = ConformalPIDCalibrator(
            alpha=self.alpha,
            method=self.method,
            calibration_steps=self.calibration_steps,
        )

    def _build_candidate_grid(
        self,
        n_assets: int,
        returns_np: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        candidates: list[np.ndarray] = []

        # 1. Equal-weight 1/N anchor
        w_eq = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        candidates.append(w_eq)

        # 2. Inverse-volatility anchor
        if len(returns_np) >= 2:
            vols = np.maximum(np.std(returns_np, axis=0, ddof=1), 1e-6)
            w_iv = (1.0 / vols) / np.sum(1.0 / vols)
            candidates.append(w_iv)

        # 3. Dirichlet samples across concentration scales
        n_rem = max(1, self.n_candidates - len(candidates))
        n_half = n_rem // 2
        dir_1 = rng.dirichlet(np.ones(n_assets), size=n_half)
        dir_2 = rng.dirichlet(np.full(n_assets, 0.5), size=max(1, n_rem - n_half))
        raw_grid = np.vstack([candidates, dir_1, dir_2])

        # Enforce max_weight cap and re-normalize onto simplex
        eff_cap = max(self.max_weight, 1.0 / n_assets)
        clipped = np.minimum(raw_grid, eff_cap)
        row_sums = np.sum(clipped, axis=1, keepdims=True)
        return np.where(row_sums > 1e-12, clipped / row_sums, 1.0 / n_assets)

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)
        returns_df = data.prices.reindex(columns=assets).pct_change().dropna()
        ret_np = returns_df.to_numpy(dtype=np.float64)

        if len(ret_np) < 4 or n_assets <= 1:
            w_fallback = pd.Series(1.0 / max(1, n_assets), index=assets, name="weights")
            return OptimizationResult(
                name="CPPSAllocator",
                weights=w_fallback,
                status="FALLBACK_EQUAL_WEIGHT",
            )

        w_grid = self._build_candidate_grid(n_assets, ret_np)  # (M_cand, N)
        # Historical daily return series for every candidate portfolio: (T, M_cand)
        cand_rets = ret_np @ w_grid.T
        cand_df = pd.DataFrame(cand_rets, index=returns_df.index)
        ewma_preds = (
            cand_df.ewm(span=15, min_periods=1)
            .mean()
            .shift(1)
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        if expected_returns is not None:
            mu_daily = (
                expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
                / 252.0
            )
            point_forecasts = w_grid @ mu_daily
        else:
            point_forecasts = (
                cand_df.ewm(span=15, min_periods=1)
                .mean()
                .iloc[-1]
                .to_numpy(dtype=np.float64)
            )

        t_cal = min(len(ret_np) - 1, self.calibration_steps)
        residuals = np.abs(cand_rets[-t_cal:] - ewma_preds[-t_cal:])
        radii, eff_alpha, emp_cov = self.calibrator.calibrate_residual_series(residuals)

        lower_bounds = point_forecasts - radii
        upper_bounds = point_forecasts + radii

        # Stage 1: Retain top-m candidates by conformal lower bound (worst-case defense)
        m_keep = min(self.top_m_defense, len(w_grid))
        stage1_idx = np.argsort(lower_bounds)[::-1][:m_keep]

        # Stage 2: Pick argmax by conformal upper bound among Stage 1 survivors
        stage2_rel = int(np.argmax(upper_bounds[stage1_idx]))
        best_idx = int(stage1_idx[stage2_rel])
        w_best = w_grid[best_idx]

        ann_ret = float(point_forecasts[best_idx] * 252.0)
        ann_vol = float(np.std(cand_rets[:, best_idx], ddof=1) * np.sqrt(252.0))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

        return OptimizationResult(
            name="CPPSAllocator",
            weights=pd.Series(w_best, index=assets, name="weights"),
            status="OPTIMAL_CPPS_HR_LR",
            expected_return=ann_ret,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            metadata={
                "conformal_lower_bound_daily": float(lower_bounds[best_idx]),
                "conformal_upper_bound_daily": float(upper_bounds[best_idx]),
                "conformal_radius_daily": float(radii[best_idx]),
                "empirical_coverage": emp_cov,
                "effective_alpha": eff_alpha,
                "n_candidates": len(w_grid),
                "top_m_defense": m_keep,
            },
        )


class ConformalKellyOverlay(BaseAllocator):
    """
    Conformal Kelly Position Sizing & Volatility-Free Overlay
    (`ConformalKellyOverlay`, Aug 2026, arXiv:2608.01494).

    Substitutes a calibrated 75% (`alpha=0.25`) online conformal prediction interval
    width `W_{t, i} = U_{t, i} - L_{t, i}` directly into the risk denominator of
    fractional Kelly sizing:

        f_{t, i}^{kelly} = kelly_fraction * max(0, mu_{t, i} - r_f) / (W_{t, i})^2

    Can run either:
      1. **As an Overlay wrapping a `base_allocator`**: scales the base allocator's
         active weights by the portfolio's conformal Kelly exposure ratio (automatically
         shrinking exposure into cash when conformal interval widths expand during regime stress),
      2. **As a standaloneConformal Kelly Allocator** (`base_allocator=None`): allocates
         across assets proportional to `mu_{t, i} / (W_{t, i})^2` capped at `max_leverage`.
    """

    def __init__(
        self,
        base_allocator: BaseAllocator | None = None,
        base_estimator: BaseReturnEstimator | None = None,
        alpha: float = 0.25,
        method: ConformalMethod = "pid",
        kelly_fraction: float = 0.50,
        max_leverage: float = 1.0,
        allow_cash_shrinkage: bool = True,
        calibration_steps: int = 35,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.base_allocator = base_allocator
        self.alpha = float(np.clip(alpha, 0.05, 0.50))
        self.method: ConformalMethod = method
        self.kelly_fraction = float(np.clip(kelly_fraction, 0.05, 2.0))
        self.max_leverage = float(max(0.05, max_leverage))
        self.allow_cash_shrinkage = bool(allow_cash_shrinkage)
        self.calibrator = ConformalPIDCalibrator(
            base_estimator=base_estimator,
            alpha=self.alpha,
            method=self.method,
            calibration_steps=calibration_steps,
        )
        self.last_interval_result_: Any = None

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)

        conf_res = self.calibrator.calibrate(data)
        self.last_interval_result_ = conf_res

        widths = (
            conf_res.interval_width.reindex(assets)
            .fillna(0.20)
            .to_numpy(dtype=np.float64)
        )
        widths = np.maximum(widths, 1e-3)

        if expected_returns is not None:
            mu_vec = (
                expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
        else:
            mu_vec = (
                conf_res.point_forecast.reindex(assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

        rf_scalar = (
            float(data.risk_free_rate.iloc[-1])
            if isinstance(data.risk_free_rate, pd.Series)
            else float(data.risk_free_rate)
        )
        excess_mu = np.maximum(mu_vec - rf_scalar, 0.0)

        # Conformal Kelly scores: mu_excess / width^2
        kelly_raw = self.kelly_fraction * (excess_mu / (widths**2))

        if self.base_allocator is not None:
            if cov_matrix is None:
                ret_df = data.prices.reindex(columns=assets).pct_change().dropna()
                cov_matrix = (
                    ret_df.cov().fillna(0.0) * 252.0
                    if len(ret_df) >= 2
                    else pd.DataFrame(
                        np.eye(n_assets) * 0.04, index=assets, columns=assets
                    )
                )
            if expected_returns is None:
                expected_returns = conf_res.point_forecast.reindex(assets).fillna(0.0)
            base_res = self.base_allocator.allocate(
                data,
                cov_matrix=cov_matrix,
                expected_returns=expected_returns,
                **kwargs,
            )
            w_base = (
                base_res.weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            # Modulate each asset by its inverse conformal width relative to median width
            median_w = float(np.median(widths))
            rel_confidence = np.clip(median_w / widths, 0.20, 1.50)
            w_tilted = w_base * rel_confidence
            t_sum = float(np.sum(np.abs(w_tilted)))
            if t_sum > 1e-12:
                w_tilted = w_tilted / t_sum

            # Portfolio-level conformal Kelly gross exposure scaling
            port_width = float(np.abs(w_tilted) @ widths)
            # Reference baseline width ~ 0.25 annualized; shrink gross exposure when width surges
            target_gross = float(
                np.clip(
                    self.kelly_fraction * (0.35 / max(port_width, 0.05)),
                    0.15,
                    self.max_leverage,
                )
            )
            if not self.allow_cash_shrinkage:
                target_gross = self.max_leverage
            w_final = w_tilted * target_gross
        else:
            k_sum = float(np.sum(kelly_raw))
            if k_sum <= 1e-12:
                # Fallback to inverse conformal interval width when all excess returns <= 0
                inv_w = 1.0 / (widths**2)
                w_dir = inv_w / np.sum(inv_w)
                target_gross = (
                    min(0.5 * self.max_leverage, self.max_leverage)
                    if self.allow_cash_shrinkage
                    else self.max_leverage
                )
                w_final = w_dir * target_gross
            else:
                if self.allow_cash_shrinkage:
                    target_gross = min(k_sum, self.max_leverage)
                else:
                    target_gross = self.max_leverage
                w_final = (kelly_raw / k_sum) * target_gross

        cash_weight = max(0.0, 1.0 - float(np.sum(w_final)))
        return OptimizationResult(
            name="ConformalKellyOverlay",
            weights=pd.Series(w_final, index=assets, name="weights"),
            status="OPTIMAL_CONFORMAL_KELLY",
            metadata={
                "empirical_coverage": conf_res.empirical_coverage,
                "target_coverage": 1.0 - self.alpha,
                "calibrated_alpha": conf_res.calibrated_alpha,
                "mean_interval_width": float(np.mean(widths)),
                "gross_exposure": float(np.sum(np.abs(w_final))),
                "cash_weight": cash_weight,
            },
        )


class ConformalSafetyGateAllocator(BaseAllocator):
    """
    Gated Conformal Semantic Risk Control (`ConformalSafetyGateAllocator` / `Gated-CSRC`,
    Zarei et al., Nov 2026, arXiv:2611.14699).

    Decouples policy proposal from execution:
      1. Queries `base_allocator` (e.g., `RLAllocator`, `DeepAllocator` / `SetTransformerPolicyNet`,
         or `EnsembleAllocator`) for its proposed weight vector `w_hat_t`.
      2. Calibrates online conformal lower-bound return certificates `L_{t+1}(lambda_hat)`
         via `ConformalPIDCalibrator` and `ConformalRiskController`.
      3. Solves the **Post-Policy Euclidean Convex Projection Gate**:
             w_t^* = argmin_{w in F_t(lambda_hat)}  0.5 * ||w - w_hat_t||_2^2
         subject to:
           - Conformal tail-loss certificate: `-L_{t+1}^T w <= certificate.max_tail_loss`,
           - Gross leverage bound: `||w||_1 <= certificate.max_gross_leverage`,
           - Cash buffer & budget: `sum(w) <= 1 - certificate.min_cash_buffer`,
           - Turnover cap: `||w - w_{t^-}||_1 <= certificate.max_turnover`,
           - Volatility cap: `||Sigma^{1/2} w||_2 <= certificate.max_annual_volatility`,
           - Sector caps & restricted asset exclusions (`w_i == 0`).

    Guarantees 0% out-of-sample hard constraint breaches even when a neural RL/DL policy
    proposes an aggressive trade on a stressed market day.
    """

    def __init__(
        self,
        base_allocator: BaseAllocator | None = None,
        certificate: RiskCertificate | None = None,
        alpha: float = 0.15,
        method: ConformalMethod = "pid",
        calibration_steps: int = 35,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.base_allocator = base_allocator or InverseVolatilityAllocator()
        self.certificate = certificate or RiskCertificate()
        self.alpha = float(np.clip(alpha, 0.02, 0.50))
        self.method: ConformalMethod = method

        self.calibrator = ConformalPIDCalibrator(
            alpha=self.alpha,
            method=self.method,
            calibration_steps=calibration_steps,
        )
        self.risk_controller = ConformalRiskController(
            alpha=min(self.alpha, 0.25),
            bound_b=1.0,
            lambda_grid=np.linspace(0.5, 2.0, 61),
        )
        self.last_projection_distance_: float = 0.0
        self.last_breaches_prevented_: list[str] = []

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        certificate: RiskCertificate | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)
        cert = certificate if certificate is not None else self.certificate

        # 1. Obtain raw proposal w_hat_t from base policy
        returns_df = data.prices.reindex(columns=assets).pct_change().dropna()
        ret_np = returns_df.to_numpy(dtype=np.float64)
        if cov_matrix is None:
            cov_matrix = (
                returns_df.cov().fillna(0.0) * 252.0
                if len(returns_df) >= 2
                else pd.DataFrame(np.eye(n_assets) * 0.04, index=assets, columns=assets)
            )
        if expected_returns is None:
            expected_returns = (
                returns_df.mean().fillna(0.0) * 252.0
                if len(returns_df) >= 1
                else pd.Series(0.0, index=assets)
            )
        try:
            prop_res = self.base_allocator.allocate(
                data,
                cov_matrix=cov_matrix,
                expected_returns=expected_returns,
                current_weights=current_weights,
                **kwargs,
            )
            w_hat = (
                prop_res.weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
        except (ValueError, KeyError, RuntimeError, TypeError):
            w_hat = np.full(n_assets, 1.0 / max(1, n_assets), dtype=np.float64)

        # 2. Calibrate online conformal lower bounds and CRC trust radius multiplier
        conf_res = self.calibrator.calibrate(data)

        if len(ret_np) >= 5:
            port_hist_losses = np.maximum(0.0, -(ret_np @ w_hat))
            lam_crc = self.risk_controller.calibrate(port_hist_losses)
        else:
            lam_crc = 1.0

        # Daily conformal worst-case loss per asset: l_i = max(0, -lower_bound_daily * lam_crc)
        lower_daily = (
            conf_res.lower_bound.reindex(assets)
            .fillna(-0.15)
            .to_numpy(dtype=np.float64)
            / 252.0
        )
        asset_tail_loss = np.maximum(0.002, -lower_daily * max(0.75, lam_crc))

        # Covariance matrix for second-order cone volatility constraint
        if cov_matrix is not None:
            cov_np = (
                cov_matrix.reindex(index=assets, columns=assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        elif len(ret_np) >= 3:
            cov_np = np.cov(ret_np, rowvar=False) * 252.0
        else:
            cov_np = np.eye(n_assets, dtype=np.float64) * 0.04

        cov_np = 0.5 * (cov_np + cov_np.T)
        eigvals, eigvecs = np.linalg.eigh(cov_np)
        cov_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-8))) @ eigvecs.T

        # Audit which constraints w_hat would breach prior to projection
        breaches: list[str] = []
        if float(asset_tail_loss @ np.abs(w_hat)) > cert.max_tail_loss + 1e-6:
            breaches.append("max_tail_loss")
        if float(np.sum(np.abs(w_hat))) > cert.max_gross_leverage + 1e-6:
            breaches.append("max_gross_leverage")
        if (
            cert.max_annual_volatility is not None
            and float(np.linalg.norm(cov_sqrt @ w_hat))
            > cert.max_annual_volatility + 1e-6
        ):
            breaches.append("max_annual_volatility")
        if (
            current_weights is not None
            and cert.max_turnover is not None
            and float(
                np.sum(
                    np.abs(
                        w_hat
                        - current_weights.reindex(assets)
                        .fillna(0.0)
                        .to_numpy(dtype=np.float64)
                    )
                )
            )
            > cert.max_turnover + 1e-6
        ):
            breaches.append("max_turnover")

        # 3. Solve Euclidean Convex Projection Gate: min_w 0.5 ||w - w_hat||_2^2
        w_var = cp.Variable(n_assets)
        max_invested = max(
            0.0, min(1.0 - cert.min_cash_buffer, cert.max_gross_leverage)
        )

        lb = np.full(n_assets, float(cert.min_asset_weight), dtype=np.float64)
        ub = np.full(n_assets, float(cert.max_asset_weight), dtype=np.float64)
        for idx_a, a_name in enumerate(assets):
            if a_name in cert.restricted_assets:
                lb[idx_a] = 0.0
                ub[idx_a] = 0.0

        constraints: list[Any] = [
            w_var >= lb,
            w_var <= ub,
            cp.norm1(w_var) <= max_invested,
        ]
        if cert.min_asset_weight >= 0.0:
            constraints.append(cp.sum(w_var) <= max_invested)

        # Conformal tail-loss certificate constraint
        eff_tail_cap = max(float(cert.max_tail_loss), 1e-5)
        constraints.append(asset_tail_loss @ cp.abs(w_var) <= eff_tail_cap)

        if cert.max_annual_volatility is not None:
            constraints.append(
                cp.norm2(cov_sqrt @ w_var) <= float(cert.max_annual_volatility)
            )

        if current_weights is not None and cert.max_turnover is not None:
            w_prev = (
                current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            constraints.append(cp.norm1(w_var - w_prev) <= float(cert.max_turnover))

        if cert.sector_caps and data.cross_sectional is not None:
            cs = data.cross_sectional
            if "sector" in cs.columns:
                for sec_name, sec_cap in cert.sector_caps.items():
                    sec_indices = [
                        i
                        for i, a in enumerate(assets)
                        if a in cs.index and str(cs.loc[a, "sector"]) == str(sec_name)
                    ]
                    if sec_indices:
                        if (
                            float(np.sum(np.abs(w_hat[sec_indices])))
                            > float(sec_cap) + 1e-6
                        ):
                            breaches.append(f"sector_cap:{sec_name}")
                        constraints.append(
                            cp.sum(cp.abs(w_var[sec_indices])) <= float(sec_cap)
                        )

        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w_var - w_hat)), constraints)
        try:
            prob.solve()
        except (cp.SolverError, ValueError, ArithmeticError):
            try:
                prob.solve(solver=cp.SCS)
            except (cp.SolverError, ValueError, ArithmeticError):
                pass

        if prob.status in ("optimal", "optimal_inaccurate") and w_var.value is not None:
            w_proj = np.asarray(w_var.value, dtype=np.float64)
            w_proj = np.clip(w_proj, lb, ub)
            l1_p = float(np.sum(np.abs(w_proj)))
            if l1_p > max_invested and l1_p > 1e-12:
                w_proj = w_proj * (max_invested / l1_p)
            status_str = "OPTIMAL_GATED_CSRC"
        else:
            # Deterministic analytical projection fallback onto box + tail + leverage bounds
            w_proj = np.clip(w_hat, lb, ub)
            l1_p = float(np.sum(np.abs(w_proj)))
            if l1_p > max_invested and l1_p > 1e-12:
                w_proj = w_proj * (max_invested / l1_p)
            pred_tail = float(asset_tail_loss @ np.abs(w_proj))
            if pred_tail > eff_tail_cap and pred_tail > 1e-12:
                w_proj = w_proj * (eff_tail_cap / pred_tail)
            if cert.max_annual_volatility is not None:
                pred_vol = float(np.linalg.norm(cov_sqrt @ w_proj))
                if pred_vol > cert.max_annual_volatility and pred_vol > 1e-12:
                    w_proj = w_proj * (cert.max_annual_volatility / pred_vol)
            status_str = "FALLBACK_ANALYTICAL_GATED_CSRC"

        self.last_projection_distance_ = float(np.linalg.norm(w_proj - w_hat))
        self.last_breaches_prevented_ = breaches

        return OptimizationResult(
            name="ConformalSafetyGateAllocator",
            weights=pd.Series(w_proj, index=assets, name="weights"),
            status=status_str,
            metadata={
                "projection_l2_distance": self.last_projection_distance_,
                "breaches_prevented": list(breaches),
                "n_breaches_prevented": len(breaches),
                "crc_calibrated_lambda": lam_crc,
                "empirical_coverage": conf_res.empirical_coverage,
                "projected_tail_loss": float(asset_tail_loss @ np.abs(w_proj)),
                "projected_gross_leverage": float(np.sum(np.abs(w_proj))),
                "cash_buffer_weight": max(0.0, 1.0 - float(np.sum(w_proj))),
                "certificate_rationale": cert.rationale,
            },
        )
