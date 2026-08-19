import pandas as pd
from typing import Union, cast, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

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
    "Standard flat fee (Retail / Small AUM)"

    def __init__(self, bps: float = 10.0) -> None:
        self.cost_pct = bps / 10000.0

    def calculate_costs(
        self,
        weights_delta: CostData,
        features_slice: dict[str, CostData] | None = None,
    ) -> CostData:
        return weights_delta.abs() * self.cost_pct


class AlmgrenChrissCostModel(BaseCostModel):
    """"""

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
