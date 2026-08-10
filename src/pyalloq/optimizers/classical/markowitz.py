import cvxpy as cp
import pandas as pd
import numpy as np
from typing import Any
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.enums import ObjectiveFunction
from pyalloq.core.data import MarketData


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
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        mu, Sigma = self._prepare_inputs(data.prices, expected_returns, cov_matrix)
        n = len(mu)

        # 1. Cast pandas values to strict float numpy arrays
        mu_vals = np.asarray(mu.values, dtype=float)
        sigma_vals_np = np.asarray(Sigma.values, dtype=float)
        Sigma_vals = cp.psd_wrap(sigma_vals_np)

        # 2. Safely extract risk-free rate
        if isinstance(data.risk_free_rate, pd.Series):
            rf = float(data.risk_free_rate.iloc[-1])
        elif data.risk_free_rate is not None:
            rf = float(data.risk_free_rate)
        else:
            rf = 0.0

        if self.objective == ObjectiveFunction.MIN_VOLATILITY:
            w = cp.Variable(n)
            # Inline the objective to prevent variable reuse errors
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w, Sigma_vals)), [cp.sum(w) == 1, w >= 0]
            )
            prob.solve()
            weights = w.value

        elif self.objective == ObjectiveFunction.MAX_SHARPE:
            mu_excess = mu_vals - rf

            if np.all(mu_excess <= 0):
                raise ValueError(
                    "All expected returns are <= risk-free rate. Max Sharpe undefined."
                )

            y = cp.Variable(n)
            # Remove .T on 1D arrays
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(y, Sigma_vals)), [mu_excess @ y == 1, y >= 0]
            )
            prob.solve()

            if y.value is not None:
                weights = y.value / np.sum(y.value)
            else:
                weights = None

        elif self.objective == ObjectiveFunction.MAX_RETURN:
            w = cp.Variable(n)
            # Extract risk_aversion from kwargs, default to 1.0
            risk_aversion = kwargs.get("risk_aversion", 1.0)

            # Remove .T on 1D arrays
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

        # Clean up matrix multiplication using @
        portfolio_return = weights @ mu_vals
        portfolio_vol = np.sqrt(weights @ sigma_vals_np @ weights)

        portfolio_sharpe = (
            (portfolio_return - rf) / portfolio_vol if portfolio_vol > 0 else 0.0
        )

        return OptimizationResult(
            weights=pd.Series(weights, index=self.tickers),
            status=prob.status.upper(),
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=portfolio_sharpe,
            metadata={
                "objective_value": prob.value,
                "solver_used": prob.solver_stats.solver_name
                if hasattr(prob, "solver_stats")
                else None,
            },
        )
