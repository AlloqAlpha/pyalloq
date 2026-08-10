import pandas as pd
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class MarketData:
    """
    Standardized Parameter Object for all financial data.
    Serves as a single source of truth across the pyalloq SDK
    """

    prices: pd.DataFrame
    # Time Series features (e.g: Volume, Factor returns, Macro Indicators, Alternative Data)
    features: dict[str, pd.DataFrame | pd.Series] = field(default_factory=dict)
    # Cross Sectional Data (e.g: Market Caps, Sector Mappings)
    cross_sectional: pd.DataFrame | None = None
    # Risk free rate (Contant or Time Series)
    risk_free_rate: pd.Series | float = 0.0

    risk_aversion: pd.Series | float = 1.0

    def __post_init__(self) -> None:
        self.validate_alignment()

    def validate_alignment(self) -> None:
        """
        Ensures Time Series Data aligns perfectly to prevent look-ahead bias.
        """
        for feat_name, df_feat in self.features.items():
            if not self.prices.index.equals(df_feat.index):
                raise ValueError(
                    f"Data misalignment: Feature: {feat_name} index does not perfectly match 'prices' index."
                )

        if isinstance(self.risk_free_rate, pd.Series):
            if not self.prices.index.equals(self.risk_free_rate.index):
                raise ValueError(
                    "Data misalignment: 'risk_free_rate' series index does not perfectly match 'prices' index."
                )

    def slice_time(
        self,
        end_date: pd.Timestamp,
        lookback: int | None = None,
    ) -> "MarketData":
        """
        Returns a new MarketData instance safely sliced for a backtest window.
        """
        sliced_prices = self.prices.loc[:end_date]
        if lookback is not None:
            sliced_prices = sliced_prices.iloc[-lookback:]

        sliced_features: dict[str, pd.DataFrame | pd.Series] = {}
        for name, feat in self.features.items():
            sliced_feat = feat.loc[:end_date]
            if lookback is not None:
                sliced_feat = sliced_feat.iloc[-lookback:]
            sliced_features[name] = sliced_feat

        sliced_rf = self.risk_free_rate
        if isinstance(self.risk_free_rate, pd.Series):
            sliced_rf = self.risk_free_rate.loc[:end_date]
            if lookback is not None:
                sliced_rf = sliced_rf.iloc[-lookback:]

        return self.__class__(
            prices=sliced_prices,
            features=sliced_features,
            cross_sectional=self.cross_sectional,
            risk_free_rate=sliced_rf,
        )
