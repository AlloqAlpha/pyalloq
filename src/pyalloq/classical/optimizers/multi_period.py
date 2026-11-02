from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import MultiPeriodOptimizationResult


def _project_capped_simplex(
    v: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    target_sum: float = 1.0,
    n_iter: int = 35,
) -> np.ndarray:
    """
    Euclidean projection onto the box-constrained simplex:
        {x in R^N : sum(x) == target_sum, low <= x_i <= high}
    via bisection on the Lagrange multiplier nu.
    """
    n = len(v)
    if n == 0:
        return v.copy()
    if n * low >= target_sum:
        return np.full(n, target_sum / n, dtype=np.float64)
    if n * high <= target_sum:
        return np.full(n, target_sum / n, dtype=np.float64)

    nu_min = float(np.min(v) - high)
    nu_max = float(np.max(v) - low)
    for _ in range(n_iter):
        nu_mid = 0.5 * (nu_min + nu_max)
        x_clipped = np.clip(v - nu_mid, low, high)
        if float(np.sum(x_clipped)) > target_sum:
            nu_min = nu_mid
        else:
            nu_max = nu_mid
    return np.clip(v - 0.5 * (nu_min + nu_max), low, high)


def _project_l1_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """Projects vector v onto the L1 ball {z : ||z||_1 <= radius}."""
    if radius <= 0.0:
        return np.zeros_like(v)
    abs_v = np.abs(v)
    if float(np.sum(abs_v)) <= radius:
        return v.copy()
    u = np.sort(abs_v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - radius))[0][-1]
    theta = float(cssv[rho_idx] - radius) / float(rho_idx + 1)
    return np.sign(v) * np.maximum(abs_v - theta, 0.0)


def soft_threshold(v: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    """
    Exact closed-form soft-thresholding proximal operator for L1 regularization:
        S(v; lambda) = sign(v) * max(|v| - lambda, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


class MultiPeriodMPCAllocator(BaseAllocator):
    """
    Model Predictive Control (MPC) Multi-Period Portfolio Optimizer (Boyd et al., 2017;
    Gârleanu & Pedersen, 2013; Rosenberg et al., 2016).

    Solves for a finite-horizon sequence of post-trade weights X = [x_{t+1}, ..., x_{t+H}]
    and trade vectors Z = [z_t, ..., z_{t+H-1}] with linearized dynamics x_{tau+1} = x_tau + z_tau:

        Maximize sum_{tau=t}^{t+H-1} beta^{tau-t} [
            mu_tau^T x_{tau+1}
            - gamma * x_{tau+1}^T Sigma_tau x_{tau+1}
            - phi_hold(x_{tau+1})
            - phi_trade(z_tau)
            - permanent_impact(x_tau, z_tau)
        ]

    Supports:
    - Proportional Transaction Costs (p = 1): L1 bid-ask & commission (no-trade region).
    - Market Impact Power Cone (p = 3/2): |z_{tau, i}|^{3/2} child-order splitting.
    - Quadratic Temporary & Permanent Impact (p = 2, Gârleanu-Pedersen state-space).
    - Dual Solver Backends:
        * `backend="cvxpy"`: Conic interior-point formulation (Rotated Quadratic & Power Cones).
        * `backend="admm"`: Alternating Direction Method of Multipliers with closed-form
          Soft-Thresholding (Section 3.1) for fast L1-constrained trajectory optimization.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        horizon: int = 5,
        discount_factor: float = 0.99,
        alpha_decay: float = 0.15,
        risk_aversion: float | None = None,
        proportional_cost_bps: float | dict[str, float] | pd.Series = 10.0,
        market_impact_coeff: float = 0.0,
        quadratic_cost_coeff: float = 0.0,
        permanent_impact_coeff: float = 0.0,
        short_holding_bps: float = 0.0,
        max_turnover_per_step: float | None = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        terminal_weights: pd.Series | dict[str, float] | None = None,
        backend: str = "cvxpy",
        admm_rho: float = 2.0,
        admm_max_iter: int = 250,
        admm_tol: float = 1e-5,
        periods_per_year: float = 52.0,
        regime_gate: Any | None = None,
    ) -> None:
        super().__init__(tickers or [])
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        if backend not in ("cvxpy", "admm"):
            raise ValueError("`backend` must be either 'cvxpy' or 'admm'.")

        self.horizon = horizon
        self.discount_factor = discount_factor
        self.alpha_decay = alpha_decay
        self.risk_aversion = risk_aversion
        self.proportional_cost_bps = proportional_cost_bps
        self.market_impact_coeff = market_impact_coeff
        self.quadratic_cost_coeff = quadratic_cost_coeff
        self.permanent_impact_coeff = permanent_impact_coeff
        self.short_holding_bps = short_holding_bps
        self.max_turnover_per_step = max_turnover_per_step
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.terminal_weights = terminal_weights
        self.backend = backend
        self.admm_rho = admm_rho
        self.admm_max_iter = admm_max_iter
        self.admm_tol = admm_tol
        self.periods_per_year = max(float(periods_per_year), 1.0)
        self.regime_gate = regime_gate

    def _build_horizon_forecasts(
        self,
        assets: list[str],
        expected_returns: pd.Series | pd.DataFrame | None,
        data: MarketData,
    ) -> np.ndarray:
        """Constructs an (H, N) matrix of multi-stage return forecasts."""
        n_assets = len(assets)
        if isinstance(expected_returns, pd.DataFrame):
            df_aligned = expected_returns.reindex(columns=assets).fillna(0.0)
            arr = df_aligned.to_numpy(dtype=np.float64)
            if len(arr) >= self.horizon:
                return arr[: self.horizon]
            pad_rows = np.repeat(arr[-1:, :], self.horizon - len(arr), axis=0)
            return np.vstack([arr, pad_rows])

        if isinstance(expected_returns, pd.Series):
            mu_0 = (
                expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
        else:
            rets = data.prices.reindex(columns=assets).pct_change().dropna()
            mu_0 = (
                rets.mean().to_numpy(dtype=np.float64) * 252.0
                if len(rets) > 0
                else np.zeros(n_assets, dtype=np.float64)
            )

        persistence = 1.0 - self.alpha_decay
        mu_matrix = np.zeros((self.horizon, n_assets), dtype=np.float64)
        for tau in range(self.horizon):
            mu_matrix[tau] = (persistence**tau) * mu_0
        return mu_matrix

    def _resolve_kappa1(self, assets: list[str]) -> np.ndarray:
        if isinstance(self.proportional_cost_bps, (int, float)):
            return np.full(
                len(assets),
                float(self.proportional_cost_bps) / 10000.0,
                dtype=np.float64,
            )
        s = pd.Series(self.proportional_cost_bps, dtype=float).reindex(assets)
        mean_val = float(s.mean()) if pd.notna(s.mean()) else 0.0
        return s.fillna(mean_val).to_numpy(dtype=np.float64) / 10000.0

    def _resolve_impact_weights(
        self, assets: list[str], cov_np: np.ndarray, data: MarketData
    ) -> np.ndarray:
        """
        Computes asset-specific liquidity weights b_i for p=3/2 and p=2 market impact,
        scaling with asset volatility and inverse square-root volume when available.
        """
        vols = np.sqrt(np.maximum(np.diag(cov_np), 1e-8))
        vol_norm = vols / (np.mean(vols) + 1e-8)
        if data.features and "volume" in data.features:
            vol_df = data.features["volume"].reindex(columns=assets)
            adv = vol_df.iloc[-20:].mean().to_numpy(dtype=np.float64)
            adv = np.maximum(adv, 1.0)
            inv_liq = (np.mean(adv) / adv) ** 0.5
            inv_liq = np.clip(inv_liq, 0.2, 5.0)
            return vol_norm * inv_liq
        return vol_norm

    def _solve_cvxpy(
        self,
        mu_h: np.ndarray,
        cov_np: np.ndarray,
        x_init: np.ndarray,
        gamma: float,
        kappa1: np.ndarray,
        impact_b: np.ndarray,
        x_term: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        h_steps, n_assets = mu_h.shape
        X = cp.Variable((h_steps, n_assets))
        Z = cp.Variable((h_steps, n_assets))

        Sigma_psd = cp.psd_wrap(cov_np)
        short_fee = self.short_holding_bps / 10000.0

        constraints: list[Any] = []
        obj_terms: list[Any] = []
        q_perm: Any = np.zeros(n_assets, dtype=np.float64)

        for tau in range(h_steps):
            beta_tau = self.discount_factor**tau
            x_prev = x_init if tau == 0 else X[tau - 1]
            x_curr = X[tau]
            z_curr = Z[tau]

            # Linearized portfolio dynamics: x_{tau+1} = x_tau + z_tau
            constraints.append(x_curr == x_prev + z_curr)
            constraints.append(cp.sum(x_curr) == 1.0)
            constraints.append(x_curr >= self.min_weight)
            constraints.append(x_curr <= self.max_weight)

            if self.max_turnover_per_step is not None:
                constraints.append(cp.norm1(z_curr) <= self.max_turnover_per_step)

            # Expected return minus quadratic variance risk
            stage_util = mu_h[tau] @ x_curr - gamma * cp.quad_form(x_curr, Sigma_psd)

            # Holding cost (short borrow fees)
            if short_fee > 0.0 and self.min_weight < 0.0:
                stage_util = stage_util - short_fee * cp.sum(cp.pos(-x_curr))

            # 1. Proportional transaction cost (p = 1)
            if np.any(kappa1 > 0.0):
                stage_util = stage_util - kappa1 @ cp.abs(z_curr)

            # 2. Power-cone market impact cost (p = 3/2)
            if self.market_impact_coeff > 0.0:
                impact_vec = self.market_impact_coeff * impact_b
                stage_util = stage_util - impact_vec @ cp.power(cp.abs(z_curr), 1.5)

            # 3. Quadratic temporary + Gârleanu-Pedersen permanent impact (p = 2)
            if self.quadratic_cost_coeff > 0.0:
                quad_weights = 0.5 * self.quadratic_cost_coeff * impact_b
                stage_util = stage_util - quad_weights @ cp.square(z_curr)

            if self.permanent_impact_coeff > 0.0:
                q_perm = (1.0 - self.alpha_decay) * q_perm + z_curr
                perm_weights = 0.5 * self.permanent_impact_coeff * impact_b
                stage_util = stage_util - perm_weights @ cp.square(q_perm)

            obj_terms.append(beta_tau * stage_util)

        if x_term is not None:
            constraints.append(X[h_steps - 1] == x_term)

        prob = cp.Problem(cp.Maximize(cp.sum(obj_terms)), constraints)
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver=cp.SCS)

        if prob.status in ("optimal", "optimal_inaccurate") and X.value is not None:
            x_val = np.asarray(X.value, dtype=np.float64)
            z_val = (
                np.asarray(Z.value, dtype=np.float64)
                if Z.value is not None
                else np.vstack([x_val[0] - x_init, np.diff(x_val, axis=0)])
            )
            return x_val, z_val, prob.status.upper(), float(prob.value or 0.0)

        # Fallback: hold x_init
        x_fallback = np.tile(x_init, (h_steps, 1))
        z_fallback = np.zeros_like(x_fallback)
        return x_fallback, z_fallback, str(prob.status).upper(), 0.0

    def _solve_admm(
        self,
        mu_h: np.ndarray,
        cov_np: np.ndarray,
        x_init: np.ndarray,
        gamma: float,
        kappa1: np.ndarray,
        impact_b: np.ndarray,
        x_term: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        """
        Alternating Direction Method of Multipliers (ADMM) with closed-form Soft-Thresholding
        (Section 3.1) for multi-period portfolio optimization under L1 & quadratic costs.
        """
        h_steps, n_assets = mu_h.shape
        rho = self.admm_rho

        X = np.tile(x_init, (h_steps, 1)).astype(np.float64)
        Y = np.zeros((h_steps, n_assets), dtype=np.float64)
        U = np.zeros((h_steps, n_assets), dtype=np.float64)

        quad_lambda = self.quadratic_cost_coeff * impact_b
        lipschitz_cov = float(np.linalg.norm(cov_np, ord=2))

        for _it in range(self.admm_max_iter):
            X_old = X.copy()
            Y_old = Y.copy()

            # 1. X-Update: Block coordinate proximal gradient + simplex projection per stage s
            for s in range(h_steps):
                if x_term is not None and s == h_steps - 1:
                    X[s] = x_term.copy()
                    continue

                beta_s = self.discount_factor**s
                x_prev = x_init if s == 0 else X[s - 1]
                x_curr = X[s]

                # Smooth gradient of stage s + coupling to s+1
                grad = beta_s * (
                    2.0 * gamma * (cov_np @ x_curr)
                    - mu_h[s]
                    + quad_lambda * (x_curr - x_prev)
                )
                grad = grad + rho * (x_curr - x_prev - Y[s] + U[s])

                curv = (
                    2.0 * beta_s * gamma * lipschitz_cov
                    + float(np.max(quad_lambda))
                    + rho
                )
                if s < h_steps - 1:
                    beta_next = self.discount_factor ** (s + 1)
                    grad = (
                        grad
                        - beta_next * quad_lambda * (X[s + 1] - x_curr)
                        - rho * (X[s + 1] - x_curr - Y[s + 1] + U[s + 1])
                    )
                    curv += float(np.max(quad_lambda)) + rho

                step_eta = 1.0 / max(curv, 1e-6)
                X[s] = _project_capped_simplex(
                    x_curr - step_eta * grad,
                    low=self.min_weight,
                    high=self.max_weight,
                    target_sum=1.0,
                )

            # 2. Y-Update: Closed-Form Soft-Thresholding + optional L1 turnover ball projection
            for s in range(h_steps):
                beta_s = self.discount_factor**s
                x_prev = x_init if s == 0 else X[s - 1]
                dx_s = X[s] - x_prev
                v_s = dx_s + U[s]
                thresh_s = (beta_s * kappa1) / rho
                y_soft = soft_threshold(v_s, thresh_s)
                if self.max_turnover_per_step is not None:
                    y_soft = _project_l1_ball(y_soft, self.max_turnover_per_step)
                Y[s] = y_soft

            # 3. Dual U-Update & Residual Convergence Check
            primal_res = 0.0
            for s in range(h_steps):
                x_prev = x_init if s == 0 else X[s - 1]
                r_s = (X[s] - x_prev) - Y[s]
                U[s] = U[s] + r_s
                primal_res += float(np.sum(r_s**2))

            dual_res = float(np.sum((rho * (Y - Y_old)) ** 2))
            max_change = float(np.max(np.abs(X - X_old)))
            if max(np.sqrt(primal_res), np.sqrt(dual_res), max_change) < self.admm_tol:
                break

        # Reconcile exact feasible sequence from soft-thresholded trades when inside no-trade zone
        for s in range(h_steps):
            x_prev = x_init if s == 0 else X[s - 1]
            if float(np.sum(np.abs(Y[s]))) < 1e-7 and (
                x_term is None or s < h_steps - 1
            ):
                X[s] = x_prev.copy()

        Z = np.vstack([X[0] - x_init, np.diff(X, axis=0)])
        obj_val = 0.0
        for s in range(h_steps):
            beta_s = self.discount_factor**s
            obj_val += beta_s * float(
                mu_h[s] @ X[s]
                - gamma * (X[s] @ cov_np @ X[s])
                - kappa1 @ np.abs(Z[s])
                - 0.5 * (quad_lambda @ (Z[s] ** 2))
            )
        return X, Z, "OPTIMAL_ADMM", obj_val

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | pd.DataFrame | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> MultiPeriodOptimizationResult:
        assets = list(cov_matrix.columns)
        n_assets = len(assets)
        cov_np = np.asarray(cov_matrix.values, dtype=np.float64)

        if self.risk_aversion is not None:
            gamma = float(self.risk_aversion)
        elif isinstance(data.risk_aversion, pd.Series):
            gamma = float(data.risk_aversion.iloc[-1])
        else:
            gamma = float(data.risk_aversion)

        if current_weights is not None:
            x_init = (
                current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            if np.sum(np.abs(x_init)) < 1e-12:
                x_init = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        else:
            x_init = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

        x_term: np.ndarray | None = None
        if self.terminal_weights is not None:
            x_term = (
                pd.Series(self.terminal_weights, dtype=float)
                .reindex(assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

        mu_h = self._build_horizon_forecasts(assets, expected_returns, data)
        mu_stage = mu_h / self.periods_per_year
        cov_stage = cov_np / self.periods_per_year
        kappa1 = self._resolve_kappa1(assets)
        impact_b = self._resolve_impact_weights(assets, cov_np, data)

        if self.backend == "admm":
            x_traj, z_traj, status, obj_val = self._solve_admm(
                mu_stage, cov_stage, x_init, gamma, kappa1, impact_b, x_term
            )
        else:
            x_traj, z_traj, status, obj_val = self._solve_cvxpy(
                mu_stage, cov_stage, x_init, gamma, kappa1, impact_b, x_term
            )

        # Optional Physics-Informed Thermodynamic Regime Gate (Section 8.2)
        gate_scale = 1.0
        if self.regime_gate is not None and hasattr(self.regime_gate, "compute_gate"):
            gate_scale = float(self.regime_gate.compute_gate(data))
            if gate_scale < 1.0:
                x_traj[0] = x_init + gate_scale * (x_traj[0] - x_init)
                z_traj[0] = x_traj[0] - x_init

        w_immediate = pd.Series(x_traj[0], index=assets, name="weights")
        stage_idx = pd.Index(
            [f"t+{tau + 1}" for tau in range(self.horizon)], name="Stage"
        )
        df_w_traj = pd.DataFrame(x_traj, index=stage_idx, columns=assets)
        df_z_traj = pd.DataFrame(z_traj, index=stage_idx, columns=assets)

        exp_ret = float(mu_h[0] @ x_traj[0])
        vol = float(np.sqrt(max(float(x_traj[0].T @ cov_np @ x_traj[0]), 0.0)))

        return MultiPeriodOptimizationResult(
            name=f"MultiPeriodMPC_{self.backend.upper()}",
            weights=w_immediate,
            status=status,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=exp_ret / vol if vol > 1e-12 else 0.0,
            weights_trajectory=df_w_traj,
            trades_trajectory=df_z_traj,
            metadata={
                "objective_value": obj_val,
                "backend": self.backend,
                "horizon": self.horizon,
                "first_step_turnover": float(np.sum(np.abs(z_traj[0]))),
                "total_planned_turnover": float(np.sum(np.abs(z_traj))),
                "regime_gate_scale": gate_scale,
            },
        )
