import pandas as pd
from abc import ABC, abstractmethod


class BaseCostModel(ABC):
    @abstractmethod
    def calculate_costs(
        self,
        weights_delta: pd.Series,
        features_slice: dict[str, pd.Series] | None = None,
    ) -> pd.Series:
        """Returns the cost in percentage terms for each asset."""
        ...


class FlatBpsCostModel(BaseCostModel):
    "Standard flat fee (Retail / Small AUM)"

    def __init__(self, bps: float = 10.0) -> None:
        self.cost_pct = bps / 10000.0

    def calculate_costs(
        self,
        weights_delta: pd.Series,
        features_slice: dict[str, pd.Series] | None = None,
    ) -> pd.Series:
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
        weights_delta: pd.Series,
        features_slice: dict[str, pd.Series] | None = None,
    ) -> pd.Series:
        if features_slice is None or "volume" not in features_slice:
            raise ValueError(
                "Almgren-Chriss requires 'volume' in data.features['volume']."
            )

        # Non-linear market impact math based on trade size relative to daily volume
        trade_dollar_size = weights_delta.abs() * self.aum
        daily_volume = features_slice["volume"]

        # Fixed Spread + Impact Cost (Gamma * sqrt(Trade Size / Volume))
        impact_cost = self.gamma * (trade_dollar_size / daily_volume).pow(0.5)
        total_cost_pct = (self.spread + impact_cost) * weights_delta.abs()

        return total_cost_pct
