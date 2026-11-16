from __future__ import annotations

from typing import Any, Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult

from pyalloq.rl.thermodynamics import WassersteinRegimeGate


class WassersteinDROAllocator(BaseAllocator):
    """
    Wasserstein Distributionally Robust Portfolio Optimizer (`WassersteinDROAllocator`).

    Constructs optimal portfolio weights against the worst-case return distribution `Q`
    within a Wasserstein ambiguity ball `B_eps(P_hat)` around the empirical or synthetic
    reference measure `P_hat`, optionally intersected with Fonseca & Junca (2021) linear
    expected-value constraints `E_Q[G r] = q`:

        P_amb = { Q : W_p(Q, P_hat) <= eps_t,  E_Q[G r] = q }

    Supported Objectives:
    ---------------------
    1. **`ObjectiveFunction.WASSERSTEIN_KELLY` (Li, 2023)**:
       Maximizes worst-case expected geometric log-growth (with optional fractional Kelly
       multiplier `kelly_fraction in (0, 1]`):
           max_{w, lambda}  sum_{i=1}^M p_i ln(1 + eta * r_tilde_i^T w)
                            - eps_t * eta * ||w - G^T lambda||_*
    2. **`ObjectiveFunction.WASSERSTEIN_CVAR` (Esfahani & Kuhn, 2018; Fonseca & Junca, 2021)**:
       Minimizes worst-case Conditional Value-at-Risk at confidence `beta in (0, 1)`:
           min_{w, alpha, s >= 0, lambda}  alpha + (1 / (1 - beta)) * (p^T s + eps_t ||w - G^T lambda||_*)
           subject to: s_i >= -r_tilde_i^T w - alpha
    3. **`ObjectiveFunction.WASSERSTEIN_MEAN_RISK`**:
       Maximizes worst-case expected return minus worst-case CVaR / variance penalty:
           max_{w, lambda}  mu_tilde^T w - eps_t ||w - G^T lambda||_* - risk_aversion * CVaR_beta^{DRO}(w)

    Ground Metrics & Dual Norms (`||u||_*`):
    ----------------------------------------
    - `"mahalanobis"`: Transport cost `d(r, r_hat) = ||Sigma^{-1/2}(r - r_hat)||_2`,
      yielding geometry-aware dual norm `||Sigma^{1/2} u||_2`.
    - `"l2"`: Euclidean transport cost, yielding dual norm `||u||_2`.
    - `"l1"`: L1 transport cost, yielding dual norm `||u||_inf`.

    Dynamic Optimal-Transport Radius Calibration (`radius_mode`):
    -------------------------------------------------------------
    - `"regime_adaptive"`: Queries `WassersteinRegimeGate` to compute the live Bures-Wasserstein
      distance `W_2(N_fast, N_slow)` and sets `eps_t = epsilon + regime_sensitivity * W_2`.
    - `"sample_scaling"`: Scales radius by finite-sample concentration `eps_t = epsilon / sqrt(M)`.
    - `"fixed"`: Uses static `eps_t = epsilon`.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        objective: ObjectiveFunction = ObjectiveFunction.WASSERSTEIN_KELLY,
        epsilon: float = 0.005,
        ground_metric: Literal["mahalanobis", "l2", "l1"] = "mahalanobis",
        radius_mode: Literal[
            "regime_adaptive", "sample_scaling", "fixed"
        ] = "regime_adaptive",
        regime_sensitivity: float = 0.25,
        regime_gate: WassersteinRegimeGate | None = None,
        kelly_fraction: float = 1.0,
        beta: float = 0.95,
        risk_aversion: float = 1.0,
        target_return: float | None = None,
        view_matrix: np.ndarray | pd.DataFrame | None = None,
        view_targets: np.ndarray | pd.Series | None = None,
        view_uncertainty: float = 1e-2,
        view_targets_are_annualized: bool = False,
        weight_bounds: tuple[float, float] | dict[str, tuple[float, float]] = (
            0.0,
            1.0,
        ),
        market_neutral: bool = False,
        max_leverage: float | None = None,
        max_turnover: float | None = None,
        l2_gamma: float = 0.0,
    ) -> None:
        super().__init__(tickers or [])
        if not (0.0 < beta < 1.0):
            raise ValueError("`beta` confidence level must be strictly in (0, 1).")
        if epsilon < 0.0:
            raise ValueError("`epsilon` Wasserstein radius must be non-negative.")
        if ground_metric not in ("mahalanobis", "l2", "l1"):
            raise ValueError(
                "`ground_metric` must be one of 'mahalanobis', 'l2', or 'l1'."
            )
        if radius_mode not in ("regime_adaptive", "sample_scaling", "fixed"):
            raise ValueError(
                "`radius_mode` must be one of 'regime_adaptive', 'sample_scaling', or 'fixed'."
            )

        self.objective = objective
        self.epsilon = float(epsilon)
        self.ground_metric = ground_metric
        self.radius_mode = radius_mode
        self.regime_sensitivity = float(max(0.0, regime_sensitivity))
        self.regime_gate = regime_gate or WassersteinRegimeGate()
        self.kelly_fraction = float(np.clip(kelly_fraction, 0.05, 2.0))
        self.beta = float(beta)
        self.risk_aversion = float(max(0.0, risk_aversion))
        self.target_return = target_return
        self.view_matrix = view_matrix
        self.view_targets = view_targets
        self.view_uncertainty = float(max(1e-6, view_uncertainty))
        self.view_targets_are_annualized = bool(view_targets_are_annualized)
        self.weight_bounds = weight_bounds
        self.market_neutral = bool(market_neutral)
        self.max_leverage = max_leverage
        self.max_turnover = max_turnover
        self.l2_gamma = float(max(0.0, l2_gamma))

    def _resolve_bounds(self, assets: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(assets)
        if isinstance(self.weight_bounds, tuple):
            low, high = float(self.weight_bounds[0]), float(self.weight_bounds[1])
            if self.market_neutral and low >= 0.0:
                low = -high
            return np.full(n, low, dtype=np.float64), np.full(n, high, dtype=np.float64)

        lb = np.zeros(n, dtype=np.float64)
        ub = np.ones(n, dtype=np.float64)
        default_low = -1.0 if self.market_neutral else 0.0
        for idx, asset in enumerate(assets):
            b = self.weight_bounds.get(asset, (default_low, 1.0))
            lb[idx] = float(b[0])
            ub[idx] = float(b[1])
        return lb, ub

    def _extract_atoms_and_probabilities(
        self,
        data: MarketData | ScenarioMarketData,
        assets: list[str],
        scenarios: ScenarioMarketData | None = None,
        scenario_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None, float]:
        """
        Extracts single-period return atoms `R` of shape `(M, N)`, reference probabilities
        `p` of shape `(M,)`, optional historical returns DataFrame (for regime gating),
        and the annualization factor.
        """
        if isinstance(data, ScenarioMarketData):
            scenarios = data
            hist_df = None
            scale_factor = 252.0
        else:
            hist_df = data.prices.reindex(columns=assets).pct_change().dropna()
            if len(hist_df) == 0:
                raise ValueError(
                    "MarketData must contain at least 2 price rows to compute returns."
                )
            scale_factor = self._resolve_annualization_factor(hist_df)

        if scenarios is not None:
            # Map 3D (M_paths, T_horizon, N_assets) into M horizon-averaged daily return atoms
            col_indices = [
                scenarios.asset_names.index(a)
                for a in assets
                if a in scenarios.asset_names
            ]
            if len(col_indices) != len(assets):
                raise ValueError(
                    "ScenarioMarketData asset_names do not cover all requested assets."
                )
            scen_rets = np.asarray(
                scenarios.returns[:, :, col_indices], dtype=np.float64
            )
            R = np.mean(scen_rets, axis=1)  # (M, N)
            m_atoms = len(R)
        else:
            assert hist_df is not None
            R = hist_df.to_numpy(dtype=np.float64)
            m_atoms = len(R)

        if scenario_probs is not None and len(scenario_probs) == m_atoms:
            probs = np.asarray(scenario_probs, dtype=np.float64)
            probs = np.maximum(probs, 0.0)
            p_sum = float(np.sum(probs))
            probs = (
                probs / p_sum
                if p_sum > 0.0
                else np.full(m_atoms, 1.0 / m_atoms, dtype=np.float64)
            )
        else:
            probs = np.full(m_atoms, 1.0 / m_atoms, dtype=np.float64)

        return R, probs, hist_df, scale_factor

    def _compute_effective_epsilon(
        self,
        R: np.ndarray,
        hist_df: pd.DataFrame | None,
        assets: list[str],
        override_eps: float | None = None,
    ) -> tuple[float, float]:
        base_eps = self.epsilon if override_eps is None else float(override_eps)
        m_atoms = len(R)

        if hist_df is not None and len(hist_df) >= 5:
            diag = self.regime_gate.evaluate(hist_df)
            w2_dist = float(diag.wasserstein_distance)
        elif m_atoms >= 6:
            tmp_df = pd.DataFrame(R, columns=assets)
            diag = self.regime_gate.evaluate(tmp_df)
            w2_dist = float(diag.wasserstein_distance)
        else:
            w2_dist = 0.0

        if self.radius_mode == "fixed":
            return base_eps, w2_dist
        if self.radius_mode == "sample_scaling":
            return base_eps / float(np.sqrt(max(1, m_atoms))), w2_dist
        # "regime_adaptive"
        return base_eps + self.regime_sensitivity * w2_dist, w2_dist

    def _resolve_views(
        self,
        assets: list[str],
        scale_factor: float,
        kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        v_mat = kwargs.get("view_matrix", self.view_matrix)
        v_tgt = kwargs.get("view_targets", self.view_targets)
        if v_mat is None or v_tgt is None:
            return None

        if isinstance(v_mat, pd.DataFrame):
            G = v_mat.reindex(columns=assets).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            G = np.asarray(v_mat, dtype=np.float64)
            if G.ndim == 1:
                G = G[None, :]

        if isinstance(v_tgt, pd.Series):
            q = v_tgt.to_numpy(dtype=np.float64)
        else:
            q = np.asarray(v_tgt, dtype=np.float64).reshape(-1)

        if G.shape[1] != len(assets) or G.shape[0] != len(q):
            raise ValueError(
                f"Incompatible view shapes: view_matrix {G.shape} vs view_targets {q.shape} for {len(assets)} assets."
            )

        if self.view_targets_are_annualized:
            q = q / scale_factor
        return G, q

    def allocate(
        self,
        data: MarketData | ScenarioMarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        scenarios: ScenarioMarketData | None = None,
        scenario_probs: np.ndarray | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if isinstance(data, ScenarioMarketData):
            assets = list(data.asset_names)
        elif cov_matrix is not None:
            assets = list(cov_matrix.columns)
        else:
            assets = list(data.prices.columns)
        n_assets = len(assets)

        R, probs, hist_df, scale_factor = self._extract_atoms_and_probabilities(
            data=data,
            assets=assets,
            scenarios=scenarios,
            scenario_probs=scenario_probs,
        )
        m_atoms = len(R)

        # Compute per-period PSD covariance matrix for Mahalanobis ground metric & OT view projection
        emp_mean = probs @ R
        R_centered = R - emp_mean[None, :]
        cov_daily = (R_centered.T * probs) @ R_centered
        if cov_matrix is not None:
            cov_passed = (
                cov_matrix.reindex(index=assets, columns=assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            # Detect if cov_matrix is annualized and convert to daily scale
            if np.trace(cov_passed) > 10.0 * np.trace(cov_daily) and scale_factor > 1.0:
                cov_daily = cov_passed / scale_factor
            else:
                cov_daily = cov_passed

        cov_daily = 0.5 * (cov_daily + cov_daily.T)
        eigvals, eigvecs = np.linalg.eigh(cov_daily)
        eigvals_psd = np.maximum(eigvals, 1e-8)
        cov_daily_psd = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T
        cov_sqrt = eigvecs @ np.diag(np.sqrt(eigvals_psd)) @ eigvecs.T

        # Optional shift to match user-supplied expected_returns
        if expected_returns is not None:
            mu_pass = (
                expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            if np.linalg.norm(mu_pass) > 5.0 * np.linalg.norm(emp_mean):
                mu_pass = mu_pass / scale_factor
            R = R + (mu_pass - emp_mean)[None, :]
            emp_mean = probs @ R

        # Fonseca & Junca (2021) Expected-Value Constraints:
        # 1. Minimum-W2 Optimal Transport projection of reference atoms onto {E_Q[G r] = q}
        # 2. Dual multiplier lambda in R^K absorbing Wasserstein ambiguity along row(G)
        views = self._resolve_views(assets, scale_factor, kwargs)
        if views is not None:
            G, q = views
            k_views = len(q)
            g_cov_gt = G @ cov_daily_psd @ G.T + 1e-8 * np.eye(
                k_views, dtype=np.float64
            )
            ot_shift = (
                cov_daily_psd @ G.T @ np.linalg.solve(g_cov_gt, q - (G @ emp_mean))
            )
            R_tilde = R + ot_shift[None, :]
            lam_var = cp.Variable(k_views)
        else:
            G = None
            q = None
            R_tilde = R
            lam_var = None

        # Clip extreme single-period scenario returns so log(1 + eta * r^T w) stays strictly inside domain
        R_tilde = np.clip(R_tilde, -0.45, 2.0)
        mu_tilde = probs @ R_tilde

        eff_eps, w2_regime = self._compute_effective_epsilon(
            R=R,
            hist_df=hist_df,
            assets=assets,
            override_eps=kwargs.get("epsilon"),
        )

        lb, ub = self._resolve_bounds(assets)
        w = cp.Variable(n_assets)
        constraints: list[Any] = [w >= lb, w <= ub]

        if self.market_neutral:
            constraints.append(cp.sum(w) == 0.0)
            lev = self.max_leverage if self.max_leverage is not None else 1.0
            constraints.append(cp.norm1(w) <= float(lev))
        else:
            constraints.append(cp.sum(w) == 1.0)
            if self.max_leverage is not None:
                constraints.append(cp.norm1(w) <= float(self.max_leverage))

        if self.max_turnover is not None and current_weights is not None:
            w_prev = (
                current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            constraints.append(cp.norm1(w - w_prev) <= float(self.max_turnover))

        target_ret = kwargs.get("target_return", self.target_return)
        if target_ret is not None:
            target_daily = (
                float(target_ret) / scale_factor
                if abs(float(target_ret)) > 0.05
                else float(target_ret)
            )
            constraints.append(mu_tilde @ w >= target_daily)

        # Effective Lipschitz slope vector u = w - G^T lambda
        eff_vec: Any = (
            w - G.T @ lam_var if (G is not None and lam_var is not None) else w
        )

        if self.ground_metric == "mahalanobis":
            dual_norm_expr = cp.norm2(cov_sqrt @ eff_vec)
        elif self.ground_metric == "l2":
            dual_norm_expr = cp.norm2(eff_vec)
        else:
            dual_norm_expr = cp.norm_inf(eff_vec)

        view_reg_expr: Any = (
            self.view_uncertainty * cp.sum_squares(lam_var)
            if lam_var is not None
            else 0.0
        )
        l2_reg_expr: Any = (
            self.l2_gamma * cp.sum_squares(w) if self.l2_gamma > 0.0 else 0.0
        )

        port_rets = R_tilde @ w  # (M,)

        if self.objective == ObjectiveFunction.WASSERSTEIN_KELLY:
            eta = self.kelly_fraction
            # Worst-case expected log-growth over B_{eff_eps}(P_hat)
            log_growth = probs @ cp.log(1.0 + eta * port_rets)
            dro_penalty = eff_eps * eta * dual_norm_expr
            obj_expr = log_growth - dro_penalty - view_reg_expr - l2_reg_expr
            prob = cp.Problem(cp.Maximize(obj_expr), constraints)
        elif self.objective in (
            ObjectiveFunction.WASSERSTEIN_CVAR,
            ObjectiveFunction.MIN_CVAR,
            ObjectiveFunction.MIN_VOLATILITY,
        ):
            alpha_var = cp.Variable()
            slacks = cp.Variable(m_atoms, nonneg=True)
            constraints.append(slacks >= -port_rets - alpha_var)
            cvar_dro = alpha_var + (1.0 / max(1.0 - self.beta, 1e-6)) * (
                probs @ slacks + eff_eps * dual_norm_expr
            )
            obj_expr = cvar_dro + view_reg_expr + l2_reg_expr
            prob = cp.Problem(cp.Minimize(obj_expr), constraints)
        else:
            # WASSERSTEIN_MEAN_RISK / MAX_SHARPE / MAX_RETURN
            alpha_var = cp.Variable()
            slacks = cp.Variable(m_atoms, nonneg=True)
            constraints.append(slacks >= -port_rets - alpha_var)
            cvar_dro = alpha_var + (1.0 / max(1.0 - self.beta, 1e-6)) * (
                probs @ slacks + eff_eps * dual_norm_expr
            )
            wc_mean = mu_tilde @ w - eff_eps * dual_norm_expr
            obj_expr = (
                wc_mean - self.risk_aversion * cvar_dro - view_reg_expr - l2_reg_expr
            )
            prob = cp.Problem(cp.Maximize(obj_expr), constraints)

        try:
            prob.solve()
        except (cp.SolverError, ValueError, ArithmeticError):
            try:
                prob.solve(solver=cp.SCS)
            except (cp.SolverError, ValueError, ArithmeticError) as err:
                _ = err

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_val = np.asarray(w.value, dtype=np.float64)
            if not self.market_neutral:
                w_val = np.clip(w_val, lb, ub)
                w_sum = float(np.sum(w_val))
                if w_sum > 1e-12:
                    w_val = w_val / w_sum
            status_str = str(prob.status).upper()
            dual_pen_val = (
                float(eff_eps * dual_norm_expr.value)
                if dual_norm_expr.value is not None
                else 0.0
            )
            wc_obj_val = float(prob.value) if prob.value is not None else 0.0
            lam_val = (
                np.asarray(lam_var.value, dtype=np.float64).tolist()
                if (lam_var is not None and lam_var.value is not None)
                else None
            )
        else:
            w_val = np.full(n_assets, 1.0 / max(1, n_assets), dtype=np.float64)
            status_str = str(prob.status).upper() if prob.status else "FALLBACK"
            dual_pen_val = 0.0
            wc_obj_val = 0.0
            lam_val = None

        ann_ret = float((mu_tilde @ w_val) * scale_factor)
        ann_vol = float(
            np.sqrt(max(0.0, (w_val @ cov_daily_psd @ w_val) * scale_factor))
        )
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

        return OptimizationResult(
            name="WassersteinDRO",
            weights=pd.Series(w_val, index=assets, name="weights"),
            status=status_str,
            expected_return=ann_ret,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            metadata={
                "objective": self.objective.name,
                "ground_metric": self.ground_metric,
                "radius_mode": self.radius_mode,
                "effective_epsilon": eff_eps,
                "wasserstein_regime_distance": w2_regime,
                "dual_norm_penalty": dual_pen_val,
                "worst_case_objective": wc_obj_val,
                "active_view_multipliers": lam_val,
                "n_atoms": m_atoms,
            },
        )
