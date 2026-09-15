import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Any
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.data import MarketData


class RiskBudgetingAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("RiskBudgetingAllocator requires cov_matrix")
        N = len(cov_matrix.columns)
        Sigma = cp.psd_wrap(cov_matrix.values)

        risk_budgets = data.features.get("risk_budgets")
        if risk_budgets is None:
            b_val = np.ones(N) / N
        else:
            risk_budgets = risk_budgets.loc[cov_matrix.columns]
            b_val = risk_budgets.to_numpy() / risk_budgets.sum()

        y = cp.Variable(N)
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
            metadata={"target_budgets": b_val.tolist()},
        )
