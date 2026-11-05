from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult


class MeanCVaRAllocator(BaseAllocator):
    """
    Single-period Conditional Value-at-Risk (CVaR / Expected Shortfall) Portfolio Optimizer
    using the Rockafellar & Uryasev (2000) convex linear programming formulation:

        CVaR_alpha(w) = alpha_var + (1 / ((1 - beta) * T)) * sum_{t=1}^T u_t
        subject to: u_t >= -r_t^T w - alpha_var, u_t >= 0.

    Supports:
    - `ObjectiveFunction.MIN_CVAR`: Minimize portfolio CVaR (optionally subject to `target_return`).
    - `ObjectiveFunction.EFFICIENT_RETURN`: Minimize portfolio CVaR subject to `mu^T w >= target_return`.
    - `ObjectiveFunction.MAX_RETURN`: Maximize `mu^T w - risk_aversion * CVaR_beta(w)`.
    - Custom `weight_bounds`, `market_neutral`, `max_leverage`, `max_turnover`, and `l2_gamma`.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        beta: float = 0.95,
        objective: ObjectiveFunction = ObjectiveFunction.MIN_CVAR,
        weight_bounds: tuple[float, float] | dict[str, tuple[float, float]] = (
            0.0,
            1.0,
        ),
        market_neutral: bool = False,
        max_leverage: float | None = None,
        target_return: float | None = None,
        risk_aversion: float | None = None,
        l2_gamma: float = 0.0,
        max_turnover: float | None = None,
    ) -> None:
        super().__init__(tickers or [])
        if not (0.0 < beta < 1.0):
            raise ValueError("`beta` confidence level must be strictly in (0, 1).")
        self.beta = float(beta)
        self.objective = objective
        self.weight_bounds = weight_bounds
        self.market_neutral = market_neutral
        self.max_leverage = max_leverage
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.l2_gamma = float(l2_gamma)
        self.max_turnover = max_turnover

    def _resolve_bounds(self, assets: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(assets)
        if isinstance(self.weight_bounds, tuple):
            low, high = float(self.weight_bounds[0]), float(self.weight_bounds[1])
            if self.market_neutral and low >= 0.0:
                low = -high
            return np.full(n, low, dtype=float), np.full(n, high, dtype=float)

        lb = np.zeros(n, dtype=float)
        ub = np.ones(n, dtype=float)
        default_low = -1.0 if self.market_neutral else 0.0
        for idx, asset in enumerate(assets):
            b = self.weight_bounds.get(asset, (default_low, 1.0))
            lb[idx] = float(b[0])
            ub[idx] = float(b[1])
        return lb, ub

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)
        returns_df = data.prices.reindex(columns=assets).pct_change().dropna()
        if len(returns_df) == 0:
            raise ValueError(
                "MarketData must contain at least 2 price rows to compute returns."
            )

        R = returns_df.to_numpy(dtype=float)
        t_obs = len(R)

        lb, ub = self._resolve_bounds(assets)
        w = cp.Variable(n_assets)
        alpha_var = cp.Variable()
        u = cp.Variable(t_obs, nonneg=True)

        tail_denom = max((1.0 - self.beta) * t_obs, 1e-8)
        cvar_expr = alpha_var + cp.sum(u) / tail_denom

        constraints: list[Any] = [
            w >= lb,
            w <= ub,
            u >= -(R @ w) - alpha_var,
        ]

        if self.market_neutral:
            constraints.append(cp.sum(w) == 0.0)
            lev = self.max_leverage if self.max_leverage is not None else 1.0
            constraints.append(cp.norm1(w) <= float(lev))
        else:
            constraints.append(cp.sum(w) == 1.0)
            if self.max_leverage is not None:
                constraints.append(cp.norm1(w) <= float(self.max_leverage))

        if self.max_turnover is not None and current_weights is not None:
            w_prev = current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            constraints.append(cp.norm1(w - w_prev) <= float(self.max_turnover))

        target_ret = kwargs.get("target_return", self.target_return)
        scale_factor = self._resolve_annualization_factor(returns_df)
        mu_series = (
            expected_returns.reindex(assets)
            if expected_returns is not None
            else returns_df.mean() * scale_factor
        )
        mu_vals = np.asarray(mu_series.values, dtype=float)

        if self.objective in (
            ObjectiveFunction.MIN_CVAR,
            ObjectiveFunction.MIN_VOLATILITY,
            ObjectiveFunction.EFFICIENT_RETURN,
        ):
            if (
                self.objective == ObjectiveFunction.EFFICIENT_RETURN
                or target_ret is not None
            ):
                if target_ret is None:
                    raise ValueError(
                        "target_return must be provided for EFFICIENT_RETURN objective."
                    )
                constraints.append(mu_vals @ w >= float(target_ret))

            obj_expr: Any = cvar_expr
            if self.l2_gamma > 0.0:
                obj_expr = obj_expr + self.l2_gamma * cp.sum_squares(w)
            prob = cp.Problem(cp.Minimize(obj_expr), constraints)

        elif self.objective == ObjectiveFunction.MAX_RETURN:
            if self.risk_aversion is not None:
                gamma = float(self.risk_aversion)
            elif isinstance(data.risk_aversion, pd.Series):
                gamma = float(data.risk_aversion.iloc[-1])
            else:
                gamma = float(data.risk_aversion)

            daily_mu = mu_vals / scale_factor
            util_expr: Any = (daily_mu @ w) - gamma * cvar_expr
            if self.l2_gamma > 0.0:
                util_expr = util_expr - self.l2_gamma * cp.sum_squares(w)
            prob = cp.Problem(cp.Maximize(util_expr), constraints)
        else:
            raise NotImplementedError(
                f"Objective {self.objective} is not supported by MeanCVaRAllocator."
            )

        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver=cp.SCS)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return OptimizationResult(
                name="MeanCVaR",
                weights=pd.Series(0.0, index=assets),
                status=str(prob.status).upper(),
            )

        weights = np.clip(np.asarray(w.value, dtype=float), lb, ub)
        if not self.market_neutral and abs(float(np.sum(weights))) > 1e-12:
            weights = weights / float(np.sum(weights))

        cov_np = (
            np.asarray(
                cov_matrix.reindex(index=assets, columns=assets).values, dtype=float
            )
            if cov_matrix is not None
            else np.asarray(returns_df.cov().values * scale_factor, dtype=float)
        )
        exp_ret = float(mu_vals @ weights)
        vol = float(np.sqrt(max(float(weights.T @ cov_np @ weights), 0.0)))
        realized_losses = -(R @ weights)
        var_val = float(np.quantile(realized_losses, self.beta))
        tail_losses = realized_losses[realized_losses >= var_val]
        cvar_val = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_val

        return OptimizationResult(
            name="MeanCVaR",
            weights=pd.Series(weights, index=assets),
            status="OPTIMAL_CVAR",
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=exp_ret / vol if vol > 1e-12 else 0.0,
            metadata={
                "beta": self.beta,
                "daily_var": var_val,
                "daily_cvar": cvar_val,
                "objective_value": float(prob.value or 0.0),
            },
        )


class CDaRAllocator(BaseAllocator):
    """
    Single-period Conditional Drawdown-at-Risk (CDaR) Portfolio Optimizer
    (Chekhlov, Uryasev & Zabarankin, 2005).
    Minimizes the tail expectation of uncompounded cumulative portfolio drawdowns over the sample path.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        beta: float = 0.95,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        target_return: float | None = None,
    ) -> None:
        super().__init__(tickers or [])
        if not (0.0 < beta < 1.0):
            raise ValueError("`beta` confidence level must be strictly in (0, 1).")
        self.beta = float(beta)
        self.weight_bounds = weight_bounds
        self.target_return = target_return

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
        R = returns_df.to_numpy(dtype=float)
        t_obs = len(R)

        cum_R = np.cumsum(R, axis=0)
        low, high = float(self.weight_bounds[0]), float(self.weight_bounds[1])

        w = cp.Variable(n_assets)
        peak = cp.Variable(t_obs)
        dd = cp.Variable(t_obs, nonneg=True)
        alpha_var = cp.Variable()
        u = cp.Variable(t_obs, nonneg=True)

        cum_port = cum_R @ w
        constraints: list[Any] = [
            cp.sum(w) == 1.0,
            w >= low,
            w <= high,
            peak >= cum_port,
            peak[0] >= 0.0,
            dd >= peak - cum_port,
            u >= dd - alpha_var,
        ]
        if t_obs > 1:
            constraints.append(peak[1:] >= peak[:-1])

        target_ret = kwargs.get("target_return", self.target_return)
        scale_factor = self._resolve_annualization_factor(returns_df)
        mu_series = (
            expected_returns.reindex(assets)
            if expected_returns is not None
            else returns_df.mean() * scale_factor
        )
        mu_vals = np.asarray(mu_series.values, dtype=float)
        if target_ret is not None:
            constraints.append(mu_vals @ w >= float(target_ret))

        tail_denom = max((1.0 - self.beta) * t_obs, 1e-8)
        cdar_expr = alpha_var + cp.sum(u) / tail_denom
        prob = cp.Problem(cp.Minimize(cdar_expr), constraints)
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver=cp.SCS)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return OptimizationResult(
                name="CDaR",
                weights=pd.Series(0.0, index=assets),
                status=str(prob.status).upper(),
            )

        weights = np.clip(np.asarray(w.value, dtype=float), low, high)
        if abs(float(np.sum(weights))) > 1e-12:
            weights = weights / float(np.sum(weights))

        cov_np = (
            np.asarray(
                cov_matrix.reindex(index=assets, columns=assets).values, dtype=float
            )
            if cov_matrix is not None
            else np.asarray(returns_df.cov().values * scale_factor, dtype=float)
        )
        exp_ret = float(mu_vals @ weights)
        vol = float(np.sqrt(max(float(weights.T @ cov_np @ weights), 0.0)))

        return OptimizationResult(
            name="CDaR",
            weights=pd.Series(weights, index=assets),
            status="OPTIMAL_CDAR",
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=exp_ret / vol if vol > 1e-12 else 0.0,
            metadata={
                "beta": self.beta,
                "cdar": float(prob.value or 0.0),
            },
        )
