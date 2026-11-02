from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import MultiPeriodOptimizationResult


class StochasticScenarioMPCAllocator(BaseAllocator):
    """
    Multi-Stage Stochastic Scenario MPC Allocator.
    Optimizes a multi-period weight trajectory X = [x_1, ..., x_H] directly over a 3D
    `ScenarioMarketData` tensor of shape (M_paths, T_horizon, N_assets), balancing
    scenario-weighted expected cumulative return, stage transaction costs, and
    terminal Conditional Value-at-Risk (CVaR / Expected Shortfall).
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        horizon: int = 5,
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.5,
        proportional_cost_bps: float = 10.0,
        max_turnover_per_step: float | None = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> None:
        super().__init__(tickers or [])
        self.horizon = horizon
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.proportional_cost_bps = proportional_cost_bps
        self.max_turnover_per_step = max_turnover_per_step
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(
        self,
        data: MarketData | ScenarioMarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        scenarios: ScenarioMarketData | None = None,
        scenario_probs: np.ndarray | None = None,
        **kwargs: Any,
    ) -> MultiPeriodOptimizationResult:
        if isinstance(data, ScenarioMarketData):
            scenarios = data
            assets = list(data.asset_names)
        else:
            assets = list(data.prices.columns)
        n_assets = len(assets)

        if current_weights is not None:
            x_init = (
                current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            if np.sum(np.abs(x_init)) < 1e-12:
                x_init = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        else:
            x_init = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

        if scenarios is not None:
            m_paths, s_horizon, _ = scenarios.returns.shape
            h_steps = min(self.horizon, s_horizon)
            scen_rets = np.asarray(scenarios.returns[:, :h_steps, :], dtype=np.float64)
        elif isinstance(data, MarketData):
            # Bootstrap historical blocks into synthetic (M, H, N) scenarios when none passed
            hist_rets = (
                data.prices.reindex(columns=assets).pct_change().dropna().to_numpy()
            )
            h_steps = self.horizon
            m_paths = min(60, max(10, len(hist_rets) - h_steps))
            rng = np.random.default_rng(42)
            starts = rng.integers(0, max(1, len(hist_rets) - h_steps), size=m_paths)
            scen_rets = np.stack(
                [hist_rets[idx : idx + h_steps] for idx in starts], axis=0
            )
        else:
            raise ValueError("Either MarketData or ScenarioMarketData is required.")

        if scenario_probs is not None and len(scenario_probs) == m_paths:
            probs = np.asarray(scenario_probs, dtype=np.float64)
            probs = probs / np.sum(probs)
        else:
            probs = np.full(m_paths, 1.0 / m_paths, dtype=np.float64)

        X = cp.Variable((h_steps, n_assets))
        Z = cp.Variable((h_steps, n_assets))
        var_aux = cp.Variable()
        cvar_slacks = cp.Variable(m_paths, nonneg=True)

        kappa1 = self.proportional_cost_bps / 10000.0
        constraints: list[Any] = []
        total_cost_expr: Any = 0.0

        for tau in range(h_steps):
            x_prev = x_init if tau == 0 else X[tau - 1]
            constraints.append(X[tau] == x_prev + Z[tau])
            constraints.append(cp.sum(X[tau]) == 1.0)
            constraints.append(X[tau] >= self.min_weight)
            constraints.append(X[tau] <= self.max_weight)
            if self.max_turnover_per_step is not None:
                constraints.append(cp.norm1(Z[tau]) <= self.max_turnover_per_step)
            total_cost_expr = total_cost_expr + kappa1 * cp.norm1(Z[tau])

        # Cumulative additive scenario portfolio return across H stages: shape (M,)
        path_returns_list = []
        for m in range(m_paths):
            path_r = cp.sum([scen_rets[m, tau] @ X[tau] for tau in range(h_steps)])
            path_returns_list.append(path_r)
        path_returns_vec = cp.hstack(path_returns_list) - total_cost_expr

        # Rockafellar-Uryasev CVaR formulation on scenario losses (-path_returns_vec)
        constraints.append(cvar_slacks >= -path_returns_vec - var_aux)
        cvar_loss = var_aux + (1.0 / max(self.cvar_alpha, 1e-4)) * (probs @ cvar_slacks)
        expected_horizon_return = probs @ path_returns_vec

        objective = cp.Maximize(expected_horizon_return - self.cvar_weight * cvar_loss)
        prob = cp.Problem(objective, constraints)
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
            status_str = prob.status.upper()
        else:
            x_val = np.tile(x_init, (h_steps, 1))
            z_val = np.zeros_like(x_val)
            status_str = str(prob.status).upper()

        stage_idx = pd.Index([f"t+{tau + 1}" for tau in range(h_steps)], name="Stage")
        return MultiPeriodOptimizationResult(
            name="StochasticScenarioMPC",
            weights=pd.Series(x_val[0], index=assets, name="weights"),
            status=status_str,
            weights_trajectory=pd.DataFrame(x_val, index=stage_idx, columns=assets),
            trades_trajectory=pd.DataFrame(z_val, index=stage_idx, columns=assets),
            metadata={
                "n_scenarios": m_paths,
                "horizon": h_steps,
                "first_step_turnover": float(np.sum(np.abs(z_val[0]))),
            },
        )
