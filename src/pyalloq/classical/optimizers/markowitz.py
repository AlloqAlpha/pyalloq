from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult


class MarkowitzAllocator(BaseAllocator):
    """
    Single-period Mean-Variance (Markowitz) Portfolio Optimizer with institutional
    constraint support:
    - Objectives: `MAX_SHARPE`, `MIN_VOLATILITY`, `MAX_RETURN` (quadratic utility),
      `EFFICIENT_RETURN` (min variance for target return), `EFFICIENT_RISK` (max return for target vol).
    - Constraints: custom per-asset or global `weight_bounds`, `market_neutral` (dollar-neutral),
      gross `max_leverage`, single-period `max_turnover`, `sector_lower`/`sector_upper` group bounds,
      and L2 diversification regularization (`l2_gamma`).
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        objective: ObjectiveFunction = ObjectiveFunction.MAX_SHARPE,
        weight_bounds: tuple[float, float] | dict[str, tuple[float, float]] = (
            0.0,
            1.0,
        ),
        market_neutral: bool = False,
        max_leverage: float | None = None,
        target_return: float | None = None,
        target_volatility: float | None = None,
        l2_gamma: float = 0.0,
        max_turnover: float | None = None,
        sector_mapper: dict[str, str] | pd.Series | None = None,
        sector_lower: dict[str, float] | None = None,
        sector_upper: dict[str, float] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.objective = objective
        self.weight_bounds = weight_bounds
        self.market_neutral = market_neutral
        self.max_leverage = max_leverage
        self.target_return = target_return
        self.target_volatility = target_volatility
        self.l2_gamma = float(l2_gamma)
        self.max_turnover = max_turnover
        self.sector_mapper = sector_mapper
        self.sector_lower = sector_lower
        self.sector_upper = sector_upper

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

    def _resolve_sector_constraints(
        self,
        w_var: Any,
        assets: list[str],
        data: MarketData,
        scale_var: Any | None = None,
    ) -> list[Any]:
        if not self.sector_lower and not self.sector_upper:
            return []

        mapper: pd.Series | None = None
        if isinstance(self.sector_mapper, pd.Series):
            mapper = self.sector_mapper.reindex(assets)
        elif isinstance(self.sector_mapper, dict):
            mapper = pd.Series(self.sector_mapper).reindex(assets)
        elif (
            data.cross_sectional is not None
            and "sector" in data.cross_sectional.columns
        ):
            mapper = data.cross_sectional["sector"].reindex(assets)

        if mapper is None:
            return []

        constraints: list[Any] = []
        rhs_scale = 1.0 if scale_var is None else scale_var

        if self.sector_lower:
            for sector, min_w in self.sector_lower.items():
                mask = (mapper == sector).to_numpy(dtype=bool)
                if np.any(mask):
                    constraints.append(cp.sum(w_var[mask]) >= float(min_w) * rhs_scale)

        if self.sector_upper:
            for sector, max_w in self.sector_upper.items():
                mask = (mapper == sector).to_numpy(dtype=bool)
                if np.any(mask):
                    constraints.append(cp.sum(w_var[mask]) <= float(max_w) * rhs_scale)

        return constraints

    def _build_standard_constraints(
        self,
        w: Any,
        assets: list[str],
        data: MarketData,
        current_weights: pd.Series | None = None,
    ) -> list[Any]:
        lb, ub = self._resolve_bounds(assets)
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
            w_prev = current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            constraints.append(cp.norm1(w - w_prev) <= float(self.max_turnover))

        constraints.extend(self._resolve_sector_constraints(w, assets, data))
        return constraints

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = list(cov_matrix.columns)
        n_assets = len(assets)

        sigma_vals_np = np.asarray(cov_matrix.values, dtype=float)
        sigma_vals_np = 0.5 * (sigma_vals_np + sigma_vals_np.T)
        Sigma_vals = cp.psd_wrap(sigma_vals_np)

        if isinstance(data.risk_free_rate, pd.Series):
            rf = float(data.risk_free_rate.iloc[-1])
        else:
            rf = float(data.risk_free_rate)

        target_ret = kwargs.get("target_return", self.target_return)
        target_vol = kwargs.get("target_volatility", self.target_volatility)

        lb, ub = self._resolve_bounds(assets)
        has_custom_sharpe_constraints = (
            self.market_neutral
            or self.max_leverage is not None
            or (self.max_turnover is not None and current_weights is not None)
            or self.l2_gamma > 0.0
            or np.any(lb != 0.0)
            or np.any(ub != 1.0)
            or bool(self.sector_lower)
            or bool(self.sector_upper)
        )

        weights: np.ndarray | None = None
        prob: cp.Problem

        if self.objective in (
            ObjectiveFunction.MIN_VOLATILITY,
            ObjectiveFunction.EFFICIENT_RETURN,
        ):
            w = cp.Variable(n_assets)
            constraints = self._build_standard_constraints(
                w, assets, data, current_weights=current_weights
            )
            if (
                self.objective == ObjectiveFunction.EFFICIENT_RETURN
                or target_ret is not None
            ):
                if expected_returns is None:
                    raise ValueError(
                        "expected_returns must be provided when using EFFICIENT_RETURN or target_return."
                    )
                if target_ret is None:
                    raise ValueError(
                        "target_return must be provided for EFFICIENT_RETURN objective."
                    )
                mu_vals = np.asarray(
                    expected_returns.reindex(assets).values, dtype=float
                )
                constraints.append(mu_vals @ w >= float(target_ret))

            obj_expr: Any = cp.quad_form(w, Sigma_vals)
            if self.l2_gamma > 0.0:
                obj_expr = obj_expr + self.l2_gamma * cp.sum_squares(w)

            prob = cp.Problem(cp.Minimize(obj_expr), constraints)
            try:
                prob.solve()
            except cp.SolverError:
                prob.solve(solver=cp.SCS)
            weights = np.asarray(w.value, dtype=float) if w.value is not None else None

        elif self.objective == ObjectiveFunction.EFFICIENT_RISK:
            if expected_returns is None:
                raise ValueError(
                    "expected_returns need to be provided for EFFICIENT_RISK objective."
                )
            if target_vol is None:
                raise ValueError(
                    "target_volatility must be provided for EFFICIENT_RISK objective."
                )
            mu_vals = np.asarray(expected_returns.reindex(assets).values, dtype=float)
            w = cp.Variable(n_assets)
            constraints = self._build_standard_constraints(
                w, assets, data, current_weights=current_weights
            )
            constraints.append(cp.quad_form(w, Sigma_vals) <= float(target_vol) ** 2)
            obj_expr = mu_vals @ w
            if self.l2_gamma > 0.0:
                obj_expr = obj_expr - self.l2_gamma * cp.sum_squares(w)

            prob = cp.Problem(cp.Maximize(obj_expr), constraints)
            try:
                prob.solve()
            except cp.SolverError:
                prob.solve(solver=cp.SCS)
            weights = np.asarray(w.value, dtype=float) if w.value is not None else None

        elif self.objective == ObjectiveFunction.MAX_SHARPE:
            if expected_returns is None:
                raise ValueError(
                    "expected_returns need to be provided for using MAX_SHARPE objective."
                )
            mu_vals = np.asarray(expected_returns.reindex(assets).values, dtype=float)
            mu_excess = mu_vals - (0.0 if self.market_neutral else rf)

            if np.all(mu_excess <= 0) and not self.market_neutral and np.all(lb >= 0):
                raise ValueError(
                    "All expected returns are <= risk-free rate. Max Sharpe undefined."
                )

            if not has_custom_sharpe_constraints:
                y = cp.Variable(n_assets)
                prob = cp.Problem(
                    cp.Minimize(cp.quad_form(y, Sigma_vals)),
                    [mu_excess @ y == 1, y >= 0],
                )
                try:
                    prob.solve()
                except cp.SolverError:
                    prob.solve(solver=cp.SCS)
                if y.value is not None and abs(float(np.sum(y.value))) > 1e-12:
                    weights = np.asarray(y.value, dtype=float) / float(np.sum(y.value))
            else:
                # General homogenized Cornuejols-Tutuncu transformation with auxiliary scale kappa >= 0
                y = cp.Variable(n_assets)
                kappa = cp.Variable(nonneg=True)
                homog_constraints: list[Any] = [
                    mu_excess @ y == 1.0,
                    y >= lb * kappa,
                    y <= ub * kappa,
                ]
                if self.market_neutral:
                    homog_constraints.append(cp.sum(y) == 0.0)
                    lev = self.max_leverage if self.max_leverage is not None else 1.0
                    homog_constraints.append(cp.norm1(y) <= float(lev) * kappa)
                else:
                    homog_constraints.append(cp.sum(y) == kappa)
                    if self.max_leverage is not None:
                        homog_constraints.append(
                            cp.norm1(y) <= float(self.max_leverage) * kappa
                        )

                if self.max_turnover is not None and current_weights is not None:
                    w_prev = (
                        current_weights.reindex(assets)
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    homog_constraints.append(
                        cp.norm1(y - w_prev * kappa) <= float(self.max_turnover) * kappa
                    )

                homog_constraints.extend(
                    self._resolve_sector_constraints(y, assets, data, scale_var=kappa)
                )

                obj_homog: Any = cp.quad_form(y, Sigma_vals)
                if self.l2_gamma > 0.0:
                    obj_homog = obj_homog + self.l2_gamma * cp.sum_squares(y)

                prob = cp.Problem(cp.Minimize(obj_homog), homog_constraints)
                try:
                    prob.solve()
                except cp.SolverError:
                    prob.solve(solver=cp.SCS)
                if (
                    y.value is not None
                    and kappa.value is not None
                    and float(kappa.value) > 1e-12
                ):
                    weights = np.asarray(y.value, dtype=float) / float(kappa.value)

        elif self.objective == ObjectiveFunction.MAX_RETURN:
            if expected_returns is None:
                raise ValueError(
                    "expected_returns need to be provided for using MAX_RETURN objective."
                )
            mu_vals = np.asarray(expected_returns.reindex(assets).values, dtype=float)
            w = cp.Variable(n_assets)
            if isinstance(data.risk_aversion, pd.Series):
                risk_aversion = float(data.risk_aversion.iloc[-1])
            else:
                risk_aversion = float(data.risk_aversion)

            constraints = self._build_standard_constraints(
                w, assets, data, current_weights=current_weights
            )
            utility: Any = (mu_vals @ w) - risk_aversion * cp.quad_form(w, Sigma_vals)
            if self.l2_gamma > 0.0:
                utility = utility - self.l2_gamma * cp.sum_squares(w)

            prob = cp.Problem(cp.Maximize(utility), constraints)
            try:
                prob.solve()
            except cp.SolverError:
                prob.solve(solver=cp.SCS)
            weights = np.asarray(w.value, dtype=float) if w.value is not None else None

        else:
            raise NotImplementedError(
                f"Objective: {self.objective} is not implemented."
            )

        if prob.status not in ["optimal", "optimal_inaccurate"] or weights is None:
            fallback_idx = self.tickers if self.tickers else assets
            return OptimizationResult(
                weights=pd.Series(0.0, index=fallback_idx),
                status=str(prob.status).upper(),
            )

        weights = np.clip(weights, lb, ub)
        if not self.market_neutral and abs(float(np.sum(weights))) > 1e-12:
            weights = weights / float(np.sum(weights))

        vol = float(np.sqrt(max(float(weights.T @ sigma_vals_np @ weights), 0.0)))
        exp_ret: float | None = None
        sharpe: float | None = None
        if expected_returns is not None:
            mu_arr = np.asarray(expected_returns.reindex(assets).values, dtype=float)
            exp_ret = float(mu_arr @ weights)
            sharpe = (
                (exp_ret - (0.0 if self.market_neutral else rf)) / vol
                if vol > 1e-12
                else 0.0
            )

        return OptimizationResult(
            name=f"Markowitz_{self.objective.name}",
            weights=pd.Series(weights, index=cov_matrix.index),
            status=str(prob.status).upper(),
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={
                "objective_value": prob.value,
                "solver_used": prob.solver_stats.solver_name
                if hasattr(prob, "solver_stats") and prob.solver_stats is not None
                else None,
            },
        )

    def efficient_frontier(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series,
        n_points: int = 20,
    ) -> list[OptimizationResult]:
        """
        Traces out `n_points` portfolios along the Mean-Variance Efficient Frontier
        between the Minimum Volatility portfolio return and the maximum feasible return.
        """
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")

        assets = list(cov_matrix.columns)
        mu_vals = np.asarray(expected_returns.reindex(assets).values, dtype=float)

        min_vol_alloc = MarkowitzAllocator(
            tickers=assets,
            objective=ObjectiveFunction.MIN_VOLATILITY,
            weight_bounds=self.weight_bounds,
            market_neutral=self.market_neutral,
            max_leverage=self.max_leverage,
            l2_gamma=self.l2_gamma,
            sector_mapper=self.sector_mapper,
            sector_lower=self.sector_lower,
            sector_upper=self.sector_upper,
        )
        res_min = min_vol_alloc.allocate(
            data, cov_matrix=cov_matrix, expected_returns=expected_returns
        )
        min_ret = (
            float(res_min.expected_return)
            if res_min.expected_return is not None
            else float(np.min(mu_vals))
        )

        lb, ub = self._resolve_bounds(assets)
        w_max = cp.Variable(len(assets))
        c_max = self._build_standard_constraints(w_max, assets, data)
        prob_max = cp.Problem(cp.Maximize(mu_vals @ w_max), c_max)
        try:
            prob_max.solve()
        except cp.SolverError:
            prob_max.solve(solver=cp.SCS)
        max_ret = (
            float(prob_max.value)
            if prob_max.status in ("optimal", "optimal_inaccurate")
            and prob_max.value is not None
            else float(np.max(mu_vals))
        )

        target_grid = np.linspace(min_ret, max_ret - 1e-6, n_points)
        frontier: list[OptimizationResult] = []
        for tr in target_grid:
            ef_alloc = MarkowitzAllocator(
                tickers=assets,
                objective=ObjectiveFunction.EFFICIENT_RETURN,
                weight_bounds=self.weight_bounds,
                market_neutral=self.market_neutral,
                max_leverage=self.max_leverage,
                target_return=float(tr),
                l2_gamma=self.l2_gamma,
                sector_mapper=self.sector_mapper,
                sector_lower=self.sector_lower,
                sector_upper=self.sector_upper,
            )
            res = ef_alloc.allocate(
                data, cov_matrix=cov_matrix, expected_returns=expected_returns
            )
            if "OPTIMAL" in res.status:
                frontier.append(res)
        return frontier
