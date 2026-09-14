import cvxpy as cp
import pandas as pd
import numpy as np
from typing import Any
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.data import MarketData


class MaxDiversificationAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("MaxDiversificationAllocator required cov_matrix")

        N = len(cov_matrix.columns)
        Sigma = cp.psd_wrap(cov_matrix.values)
        vols = np.sqrt(np.diag(cov_matrix.values))

        y = cp.Variable(N)
        obj = cp.Minimize(cp.quad_form(y, Sigma))
        constraints = [
            y.T @ vols == 1,  # Force weighted sum of vols == 1
            y >= 0,  # Long only
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"] or y.value is None:
            return OptimizationResult(
                weights=pd.Series(0, index=cov_matrix.columns),
                status=prob.status.upper(),
            )
        weights = y.value / np.sum(y.value)
        return OptimizationResult(
            weights=pd.Series(weights, index=cov_matrix.columns),
            status="OPTIMAL",
        )
