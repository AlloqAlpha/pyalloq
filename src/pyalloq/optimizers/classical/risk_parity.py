import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Any
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.data import MarketData


class RiskParityAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("RiskBudgetingAllocator requires cov_matrix")
        N = len(data.prices.columns)
        Sigma = cp.psd_wrap(cov_matrix.values)

        y = cp.Variable(N)

        objective = cp.Minimize(0.5 * cp.quad_form(y, Sigma) - cp.sum(cp.log(y)))

        constraints = [
            y >= 0  # non-negativity
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if prob.status not in ["optimal", "optimal_inaccurate"] or y.value is None:
            return OptimizationResult(
                weights=pd.Series(0, index=cov_matrix.columns),
                status=prob.status.upper(),
            )

        weights = y.value / np.sum(y.value)

        return OptimizationResult(
            weights=pd.Series(weights, index=cov_matrix.columns),
            status="OPTIMAL_RISK_PARITY",
        )
