from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult


class RiskParityAllocator(BaseAllocator):
    """
    Every asset contributes the exact same amount of volatility to overall portfolio.
    """

    def allocate(
        self,
        data: MarketData | None = None,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("RiskBudgetingAllocator requires cov_matrix")
        all_assets = list(cov_matrix.columns)
        diag_vals = np.diag(np.asarray(cov_matrix.values, dtype=np.float64))
        not_sentinel = (diag_vals > 0.0) & (diag_vals < 1e4)
        tradable_attr = cov_matrix.attrs.get("tradability_mask")
        if tradable_attr is not None:
            active_mask = np.asarray(tradable_attr, dtype=bool) & not_sentinel
        elif data is not None and len(data.prices) > 0:
            latest_p = data.prices.reindex(columns=all_assets).iloc[-1]
            active_mask = (
                latest_p.notna().to_numpy(dtype=bool)
                & np.isfinite(latest_p.to_numpy(dtype=np.float64))
                & not_sentinel
            )
        else:
            active_mask = not_sentinel

        if not np.all(active_mask) and np.any(active_mask):
            active_assets = [a for a, is_act in zip(all_assets, active_mask) if is_act]
            sub_cov = cov_matrix.loc[active_assets, active_assets].copy()
            sub_cov.attrs["tradability_mask"] = [True] * len(active_assets)
            sub_res = self.allocate(
                data=data,
                cov_matrix=sub_cov,
                expected_returns=expected_returns,
                **kwargs,
            )
            full_w = sub_res.weights.reindex(all_assets, fill_value=0.0)
            return OptimizationResult(
                weights=full_w,
                status=sub_res.status,
            )

        N = len(cov_matrix.columns)
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
