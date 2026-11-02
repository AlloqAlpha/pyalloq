from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union, cast

import pandas as pd

if TYPE_CHECKING:
    import torch

    CostData = Union[pd.Series, pd.DataFrame, torch.Tensor]
else:
    CostData = Union[pd.Series, pd.DataFrame, Any]


class BaseCostModel(ABC):
    @abstractmethod
    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        """Returns the cost in percentage terms for each asset."""
        ...


class FlatBpsCostModel(BaseCostModel):
    """Standard flat fee (Retail / Small AUM)."""

    def __init__(self, bps: float = 10.0) -> None:
        self.bps = bps
        self.cost_pct = bps / 10000.0

    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        return weights_delta.abs() * self.cost_pct


class ProportionalSpreadCostModel(BaseCostModel):
    """
    Asset-specific or uniform bid-ask half-spread + fixed commission cost model.
    Supports either a scalar bps, a dict mapping ticker -> full spread bps, or a Series.
    Charges `0.5 * spread_bps + commission_bps` per unit of absolute weight traded.
    """

    def __init__(
        self,
        spread_bps: float | dict[str, float] | pd.Series = 5.0,
        commission_bps: float = 0.0,
    ) -> None:
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps

    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        comm_pct = self.commission_bps / 10000.0
        if isinstance(self.spread_bps, (int, float)):
            half_spread_pct = (0.5 * float(self.spread_bps)) / 10000.0
            return weights_delta.abs() * (half_spread_pct + comm_pct)

        spread_series = pd.Series(self.spread_bps, dtype=float)
        if isinstance(weights_delta, pd.Series):
            aligned_spread = spread_series.reindex(weights_delta.index).fillna(
                float(spread_series.mean()) if len(spread_series) > 0 else 0.0
            )
            half_spread_pct_s = (0.5 * aligned_spread) / 10000.0
            return weights_delta.abs() * (half_spread_pct_s + comm_pct)

        if isinstance(weights_delta, pd.DataFrame):
            aligned_spread = spread_series.reindex(weights_delta.columns).fillna(
                float(spread_series.mean()) if len(spread_series) > 0 else 0.0
            )
            half_spread_pct_s = (0.5 * aligned_spread) / 10000.0
            return weights_delta.abs().mul(half_spread_pct_s + comm_pct, axis=1)

        # Fallback for tensor inputs
        mean_half_spread = (
            (0.5 * float(spread_series.mean())) / 10000.0
            if len(spread_series) > 0
            else 0.0
        )
        return weights_delta.abs() * (mean_half_spread + comm_pct)


class AlmgrenChrissCostModel(BaseCostModel):
    """
    Non-linear square-root market impact + fixed spread cost model (Almgren & Chriss).
    Requires 'volume' in `features_slice`.
    """

    def __init__(
        self, portfolio_aum: float, spread_bps: float = 2.0, gamma: float = 0.1
    ) -> None:
        self.aum = portfolio_aum
        self.spread = spread_bps / 10000.0
        self.gamma = gamma

    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        if features_slice is None or "volume" not in features_slice:
            raise ValueError(
                "Almgren-Chriss requires 'volume' in data.features['volume']."
            )

        delta = cast(Any, weights_delta)
        daily_volume = cast(Any, features_slice["volume"])

        # Non-linear market impact math based on trade size relative to daily volume
        trade_dollar_size = delta.abs() * self.aum

        # Safe for torch usage
        safe_volume = daily_volume + 1e-8
        safe_trade = trade_dollar_size + 1e-8

        # Fixed Spread + Impact Cost (Gamma * sqrt(Trade Size / Volume))
        impact_cost = self.gamma * (safe_trade / safe_volume).pow(0.5)  # type: ignore[operator]
        total_cost_pct = (self.spread + impact_cost) * weights_delta.abs()  # type: ignore[operator]

        return total_cost_pct


class CompositeCostModel(BaseCostModel):
    """Combines multiple BaseCostModel instances additively."""

    def __init__(self, models: Sequence[BaseCostModel]) -> None:
        if not models:
            raise ValueError("CompositeCostModel requires at least one cost model.")
        self.models = list(models)

    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        total = self.models[0].calculate_costs(weights_delta, features_slice)
        for model in self.models[1:]:
            total = total + model.calculate_costs(weights_delta, features_slice)
        return total
