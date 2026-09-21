import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class MarketData:
    """
    Standardized Parameter Object for all financial data.
    Serves as a single source of truth across the pyalloq SDK
    """

    assets: list[str] = field(default_factory=list)
    prices: pd.DataFrame
    # Time Series features (e.g: Volume, Factor returns, Macro Indicators, Alternative Data)
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Known Future Covariates (e.g., scheduled macro events, earnings call)
    future_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Cross Sectional Data (e.g: Market Caps, Sector Mappings)
    cross_sectional: pd.DataFrame | None = None
    # Risk free rate (Contant or Time Series)
    risk_free_rate: pd.Series | float = 0.0
    # Risk aversion rate (0: Most risk taking - 1: Most conservative)
    risk_aversion: pd.Series | float = 1.0

    def __post_init__(self) -> None:
        if not self.assets and self.prices is not None and not self.prices.empty:
            self.assets = list(self.prices.columns)
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

        for feat_name, df_feat in self.future_features.items():
            if not self.prices.index.equals(df_feat.index):
                raise ValueError(
                    f"Data misaligment: Future feature '{feat_name}' is missing historical dates."
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
        horizon_steps: int = 0,
    ) -> "MarketData":
        """
        Returns a new MarketData instance safely sliced for a backtest window.
        """
        sliced_prices = self.prices.loc[:end_date]
        if lookback is not None:
            sliced_prices = sliced_prices.iloc[-lookback:]

        sliced_features: dict[str, pd.DataFrame] = {}
        for name, feat in self.features.items():
            sliced_feat = feat.loc[:end_date]
            if lookback is not None:
                sliced_feat = sliced_feat.iloc[-lookback:]
            sliced_features[name] = sliced_feat

        sliced_futures: dict[str, pd.DataFrame] = {}
        for name, feat in self.future_features.items():
            if end_date in feat.index:
                loc = feat.index.get_loc(end_date)
                if not isinstance(loc, int):
                    raise ValueError(
                        f"Duplicate timestamps found in future_features['{name}']"
                        "Time series indices must be strictly unique."
                    )
                start_loc = max(0, (loc - lookback + 1)) if lookback else 0
                end_loc = loc + 1 + horizon_steps
                sliced_futures[name] = feat.iloc[start_loc:end_loc]
            else:
                sliced_futures[name] = feat.loc[:end_date]

        sliced_rf = self.risk_free_rate
        if isinstance(self.risk_free_rate, pd.Series):
            sliced_rf = self.risk_free_rate.loc[:end_date]
            if lookback is not None:
                sliced_rf = sliced_rf.iloc[-lookback:]

        return self.__class__(
            prices=sliced_prices,
            features=sliced_features,
            future_features=sliced_futures,
            cross_sectional=self.cross_sectional,
            risk_free_rate=sliced_rf,
        )

    def get_returns(self, log_returns: bool = False) -> None:
        pass


@dataclass(kw_only=True)
class ScenarioMarketData:
    """
    Core 3D tensor container for multi-path synthetic futures in PyAlloq.
    Shapes:
        prices: (M, T, N) - M paths, T time steps, N assets
        returns: (M, T, N)
    """

    prices: np.ndarray
    returns: np.ndarray
    asset_names: list[str]
    timestamps: pd.RangeIndex = field(default_factory=lambda: pd.RangeIndex(0))
    features: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_paths, self.horizon, self.n_assets = self.prices.shape
        if len(self.timestamps) == 0:
            self.timestamps = pd.RangeIndex(start=0, stop=self.horizon, step=1)

    def get_path(self, path_idx: int) -> MarketData:
        """
        Slices the 3D tensor at index `path_idx` and returns a standard 2D MarketData
        object mathematically compatible with the existing StrategyPipeline.
        """
        path_prices = pd.DataFrame(
            self.prices[path_idx], index=self.timestamps, columns=self.asset_names
        )

        path_features = {}
        for feature_name, feature_tensor in self.features.items():
            # Support for both asset-specific features (M, T, N) and exogenous macro features (M, T, 1)
            if feature_tensor.ndim == 3 and feature_tensor.shape[0] == self.n_paths:
                columns = (
                    self.asset_names
                    if feature_tensor.shape[2] == self.n_assets
                    else None
                )
                path_features[feature_name] = pd.DataFrame(
                    feature_tensor[path_idx], index=self.timestamps, columns=columns
                )

        return MarketData(prices=path_prices, features=path_features)
