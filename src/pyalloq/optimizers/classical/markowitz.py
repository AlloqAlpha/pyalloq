import cvxpy as cp
import pandas as pd
import numpy as np
from typing import Any
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.data import MarketData


class MarkowitzAllocator(BaseAllocator):
    def __init__(
        self,
        tickers: list[str],
        objective: ObjectiveFunction = ObjectiveFunction.MAX_SHARPE,
    ) -> None:
        super().__init__(tickers)
        self.objective = objective

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        N = len(cov_matrix.columns)

        sigma_vals_np = np.asarray(cov_matrix.values, dtype=float)
        Sigma_vals = cp.psd_wrap(sigma_vals_np)

        if isinstance(data.risk_free_rate, pd.Series):
            # TODO: Use whole Series instead
            rf = float(data.risk_free_rate.iloc[-1])
        else:
            rf = data.risk_free_rate

        if self.objective == ObjectiveFunction.MIN_VOLATILITY:
            w = cp.Variable(N)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w, Sigma_vals)), [cp.sum(w) == 1, w >= 0]
            )
            prob.solve()
            weights = w.value

        elif self.objective == ObjectiveFunction.MAX_SHARPE:
            if expected_returns is None:
                raise ValueError(
                    "expected_returns need to be provided for using MAX_SHARPE objective."
                )
            mu_vals = np.asarray(expected_returns.values, dtype=float)

            mu_excess = mu_vals - rf

            if np.all(mu_excess <= 0):
                raise ValueError(
                    "All expected returns are <= risk-free rate. Max Sharpe undefined."
                )

            y = cp.Variable(N)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(y, Sigma_vals)), [mu_excess @ y == 1, y >= 0]
            )
            prob.solve()

            if y.value is not None:
                weights = y.value / np.sum(y.value)
            else:
                weights = None

        elif self.objective == ObjectiveFunction.MAX_RETURN:
            if expected_returns is None:
                raise ValueError(
                    "expected_returns need to be provided for using MAX_SHARPE objective."
                )
            mu_vals = np.asarray(expected_returns.values, dtype=float)
            w = cp.Variable(N)
            risk_aversion = data.risk_aversion
            utility = (mu_vals @ w) - risk_aversion * cp.quad_form(w, Sigma_vals)
            prob = cp.Problem(cp.Maximize(utility), [cp.sum(w) == 1, w >= 0])
            prob.solve()
            weights = w.value

        else:
            raise NotImplementedError(
                f"Objective: {self.objective} is not implemented."
            )

        if prob.status not in ["optimal", "optimal_inaccurate"] or weights is None:
            return OptimizationResult(
                weights=pd.Series(0, index=self.tickers), status=prob.status.upper()
            )

        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)

        return OptimizationResult(
            weights=pd.Series(weights, index=cov_matrix.index),
            status=prob.status.upper(),
            metadata={
                "objective_value": prob.value,
                "solver_used": prob.solver_stats.solver_name
                if hasattr(prob, "solver_stats")
                else None,
            },
        )
