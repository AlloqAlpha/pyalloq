import cvxpy as cp
import pandas as pd
import numpy as np
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.enums import ObjectiveFunction

class MarkowitzAllocator(BaseAllocator):
    def allocate(
        self,
        prices: pd.DataFrame | None = None,
        expected_returns: pd.DataFrame | None = None,
        cov_matrix: pd.DataFrame | None = None,
        objective: ObjectiveFunction = ObjectiveFunction.MAX_SHARPE,
        risk_free_rate: float = 0.0,
        risk_aversion: float = 1.0,
    ) -> OptimizationResult:

        mu, Sigma = self._prepare_inputs(prices, expected_returns, cov_matrix)
        n = len(mu)
        mu_vals = mu.values

        Sigma_vals = cp.psd_wrap(Sigma.values)
        if objective == ObjectiveFunction.MIN_VOLATILITY:
            w = cp.Variable(n)
            obj = cp.Minimize(cp.quad_form(w, Sigma_vals))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(obj, constraints)
            prob.solve()
            weights = w.value
        elif objective == ObjectiveFunction.MAX_SHARPE:
            mu_excess = mu_vals - risk_free_rate

            if np.all(mu_excess <= 0):
                raise ValueError("All expected returns are <= risk-free rate. Max Sharpe undefined.")

            y = cp.Variable(n)
            obj = cp.Minimize(cp.quad_form(y, Sigma_vals))

            constraints = [mu_excess.T @ y == 1, y >= 0]
            prob = cp.Problem(obj, constraints)
            prob.solve()

            if y.value is not None:
                weights = y.value / np.sum(y.value)
            else:
                weights = None
        elif objective == ObjectiveFunction.MAX_RETURN:
            w = cp.Variable(n)
            utility = (mu_vals.T @ w) - risk_aversion * cp.quad_form(w, Sigma_vals)
            obj = cp.Maximize(utility)
            constraints = [cp.sum(w), constraints]

            prob = cp.Problem(obj, constraints)
            prob.solve()
            weights = w.value
        else:
            raise NotImplementedError(f"Objective: {objective} is not implemented.")

        if prob.status not in ["optimal", "optimal_inaccurate"] or weights is None:
            return OptimizationResult(
                weights=pd.Series(0, index=self.tickers),
                status=prob.status.upper()
            )

        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mu_vals)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma.values, weights)))
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0

        return OptimizationResult(
            weights=pd.Series(weights, index=self.tickers),
            status=prob.status.upper(),
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=portfolio_sharpe,
            metadata={
                "objective_value": prob.value,
                "solver_used": prob.solver_stats.solver_name if hasattr(prob, "solver_stats") else None
            }
        )