import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Optional
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult

class RiskBudgetingAllocator(BaseAllocator):
    def allocate(
        self,
        cov_matrix: pd.DataFrame,
        risk_budgets: pd.Series | None = None,
        **kwargs,
    ) -> OptimizationResult:
        n = len(cov_matrix)
        Sigma = cp.psd_wrap(cov_matrix.values)

        if risk_budgets is None:
            b_val = np.ones(n) / n
        else:
            risk_budgets = risk_budgets.loc[cov_matrix.columns]
            b_val = risk_budgets.values / risk_budgets.sum()

        y = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(y, Sigma) - b_val.T @ cp.log(y))
        constraints = [
            y >= 0,
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
            status="OPTIMAL_RISK_BUDGETING",
            metadata={"target_budgets": b_val.tolist()}
        )