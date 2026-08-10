import cvxpy as cp
import pandas as pd
import numpy as np
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.data import MarketData


class MaxDiversificationAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        **kwargs,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("MaxDiversificationAllocator required cov_matrix")

        N = len(data.prices.columns)
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
