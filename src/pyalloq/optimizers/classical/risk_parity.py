import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult

class RiskParityAllocator(BaseAllocator):
    def allocate(
        self,
        cov_matrix: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        n = len(cov_matrix)
        Sigma = cp.psd_wrap(cov_matrix.values)

        y = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(y, Sigma) - cp.sum(cp.log(y)))

        constraints = [
            y >= 0 # non-negativity
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if prob.status not in ["optimal", "optimal_inaccurate"] or y.value is None:
            return OptimizationResult(
                weights=pd.Series(0, index=cov_matrix.columns),
                status=prob.status.upper()
            )

        weights = y.value / np.sum(y.value)

        return OptimizationResult(
            weights=pd.Series(weights, index=cov_matrix.columns),
            status="OPTIMAL_RISK_PARITY",
        )