import numpy as np
from dataclasses import dataclass
from typing import Literal, Union

from pyalloq_core.data import ScenarioMarketData


@dataclass
class AbsoluteView:
    asset: str
    operator: Literal["<=", ">=", "=="]
    target: float


@dataclass
class RelativeView:
    asset_outperformer: str
    asset_underperformer: str
    operator: Literal["<=", ">=", "=="]
    target: float  # e.g., 0.02 means outperformer beats underperformer by 2%


ViewType = Union[AbsoluteView, RelativeView]


class ViewBuilder:
    """
    Translates human-readable market views into inequality and equality
    matrices for the EntropyPooler.
    """

    def __init__(self, scenario_data: ScenarioMarketData):
        self.asset_names = scenario_data.asset_names

        initial_prices = scenario_data.prices[:, 0, :]
        final_prices = scenario_data.prices[:, -1, :]

        # Shape: (M, N). Transpose to (N, M) so each row is an asset's scenario distribution
        self.scenario_returns = ((final_prices - initial_prices) / initial_prices).T
        self.asset_idx = {name: i for i, name in enumerate(self.asset_names)}
        self.M = scenario_data.prices.shape[0]

    def _get_asset_vector(self, asset_name: str) -> np.ndarray:
        if asset_name not in self.asset_idx:
            raise ValueError(f"Asset: {asset_name} not found in ScenarioMarketData.")
        return self.scenario_returns[self.asset_idx[asset_name], :]

    def build(
        self, views: list[ViewType]
    ) -> tuple[
        np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]:
        """
        Parses views and returns (A_ineq, b_ineq, C_eq, d_eq).
        Returns None for matrices if no corresponding constraints exist.
        """
        A_ineq, b_ineq = [], []
        C_eq, d_eq = [], []

        for view in views:
            if isinstance(view, AbsoluteView):
                condition_vector = self._get_asset_vector(view.asset)
            elif isinstance(view, RelativeView):
                vec_out = self._get_asset_vector(view.asset_outperformer)
                vec_under = self._get_asset_vector(view.asset_underperformer)
                condition_vector = vec_out - vec_under
            else:
                raise TypeError("Unsupported view type.")

            if view.operator == "==":
                C_eq.append(condition_vector)
                d_eq.append(view.target)
            elif view.operator == "<=":
                A_ineq.append(condition_vector)
                b_ineq.append(view.target)
            elif view.operator == ">=":
                A_ineq.append(-condition_vector)
                b_ineq.append(-view.target)

        A_out = np.vstack(A_ineq) if A_ineq else None
        b_out = np.array(b_ineq) if b_ineq else None
        C_out = np.vstack(C_eq) if C_eq else None
        d_out = np.array(d_eq) if d_eq else None

        return A_out, b_out, C_out, d_out
