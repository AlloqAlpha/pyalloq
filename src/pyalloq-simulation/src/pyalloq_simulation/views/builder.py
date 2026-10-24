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


@dataclass
class VolatilityView:
    """
    Second-moment view on an asset's horizon return volatility (standard deviation).
    Specify either `target_volatility` (e.g. 0.25 for 25% vol) or `multiplier`
    (e.g. 1.5 for a 50% increase over prior scenario volatility).
    """

    asset: str
    operator: Literal["<=", ">=", "=="]
    target_volatility: float | None = None
    multiplier: float | None = None

    def __post_init__(self) -> None:
        if (self.target_volatility is None) == (self.multiplier is None):
            raise ValueError(
                "Specify exactly one of `target_volatility` or `multiplier` in VolatilityView."
            )


@dataclass
class CovarianceView:
    """
    Second-moment view on the covariance or correlation between two assets.
    Specify either `target_covariance` or `target_correlation` in [-1, 1].
    """

    asset_1: str
    asset_2: str
    operator: Literal["<=", ">=", "=="]
    target_covariance: float | None = None
    target_correlation: float | None = None

    def __post_init__(self) -> None:
        if (self.target_covariance is None) == (self.target_correlation is None):
            raise ValueError(
                "Specify exactly one of `target_covariance` or `target_correlation` in CovarianceView."
            )


ViewType = Union[AbsoluteView, RelativeView, VolatilityView, CovarianceView]


class ViewBuilder:
    """
    Translates human-readable first- and second-moment market views into
    linear inequality and equality matrices for the EntropyPooler.
    """

    def __init__(self, scenario_data: ScenarioMarketData) -> None:
        self.asset_names = scenario_data.asset_names

        # Shape: (M, N). Transpose to (N, M) so each row is an asset's scenario distribution
        self.scenario_returns = (
            scenario_data.get_cumulative_returns().astype(np.float64).T
        )
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
        A_ineq: list[np.ndarray] = []
        b_ineq: list[float] = []
        C_eq: list[np.ndarray] = []
        d_eq: list[float] = []

        # Track reference means and volatilities for centering second-moment views
        ref_means: dict[str, float] = {
            name: float(np.mean(self._get_asset_vector(name)))
            for name in self.asset_names
        }
        ref_stds: dict[str, float] = {
            name: float(np.std(self._get_asset_vector(name)))
            for name in self.asset_names
        }

        for view in views:
            if isinstance(view, AbsoluteView) and view.operator == "==":
                if view.asset in self.asset_idx:
                    ref_means[view.asset] = float(view.target)

        for view in views:
            if isinstance(view, VolatilityView):
                vec = self._get_asset_vector(view.asset)
                prior_std = float(np.std(vec))
                target_vol = (
                    float(view.target_volatility)
                    if view.target_volatility is not None
                    else prior_std * float(view.multiplier)  # type: ignore[arg-type]
                )
                ref_stds[view.asset] = target_vol

        for view in views:
            if isinstance(view, AbsoluteView):
                condition_vector = self._get_asset_vector(view.asset)
                target_val = float(view.target)
            elif isinstance(view, RelativeView):
                vec_out = self._get_asset_vector(view.asset_outperformer)
                vec_under = self._get_asset_vector(view.asset_underperformer)
                condition_vector = vec_out - vec_under
                target_val = float(view.target)
            elif isinstance(view, VolatilityView):
                vec = self._get_asset_vector(view.asset)
                mean_val = ref_means[view.asset]
                prior_std = float(np.std(vec))
                target_vol = (
                    float(view.target_volatility)
                    if view.target_volatility is not None
                    else prior_std * float(view.multiplier)  # type: ignore[arg-type]
                )
                condition_vector = (vec - mean_val) ** 2
                target_val = target_vol**2
            elif isinstance(view, CovarianceView):
                vec1 = self._get_asset_vector(view.asset_1)
                vec2 = self._get_asset_vector(view.asset_2)
                m1 = ref_means[view.asset_1]
                m2 = ref_means[view.asset_2]
                condition_vector = (vec1 - m1) * (vec2 - m2)
                if view.target_covariance is not None:
                    target_val = float(view.target_covariance)
                else:
                    target_val = (
                        float(view.target_correlation)  # type: ignore[arg-type]
                        * ref_stds[view.asset_1]
                        * ref_stds[view.asset_2]
                    )
            else:
                raise TypeError("Unsupported view type.")

            if view.operator == "==":
                C_eq.append(condition_vector)
                d_eq.append(target_val)
            elif view.operator == "<=":
                A_ineq.append(condition_vector)
                b_ineq.append(target_val)
            elif view.operator == ">=":
                A_ineq.append(-condition_vector)
                b_ineq.append(-target_val)

        A_out = np.vstack(A_ineq) if A_ineq else None
        b_out = np.array(b_ineq, dtype=np.float64) if b_ineq else None
        C_out = np.vstack(C_eq) if C_eq else None
        d_out = np.array(d_eq, dtype=np.float64) if d_eq else None

        return A_out, b_out, C_out, d_out
