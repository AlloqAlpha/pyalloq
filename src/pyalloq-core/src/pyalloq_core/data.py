from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from .enums import CovariateKey, CovariateMode


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

        for feat_name, df_feat in list(self.future_features.items()):
            t_len = len(self.prices.index)
            if (
                t_len > 0
                and len(df_feat.index) > t_len
                and df_feat.index[0] < self.prices.index[0]
                and self.prices.index[0] in df_feat.index
            ):
                df_feat = df_feat.loc[self.prices.index[0] :]
                self.future_features[feat_name] = df_feat
            if len(df_feat.index) < t_len or not df_feat.index[:t_len].equals(
                self.prices.index
            ):
                raise ValueError(
                    f"Data misaligment: Future feature '{feat_name}' is missing historical dates."
                )

        if isinstance(self.risk_free_rate, pd.Series) and not self.prices.index.equals(
            self.risk_free_rate.index
        ):
            raise ValueError(
                "Data misalignment: 'risk_free_rate' series index does not perfectly match 'prices' index."
            )

        if isinstance(self.risk_aversion, pd.Series) and not self.prices.index.equals(
            self.risk_aversion.index
        ):
            raise ValueError(
                "Data misalignment: 'risk_aversion' series index does not perfectly match 'prices' index."
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
            if feat.index.has_duplicates:
                raise ValueError(
                    f"Duplicate timestamps found in future_features['{name}']"
                    "Time series indices must be strictly unique."
                )
            if len(sliced_prices.index) > 0 and sliced_prices.index[0] in feat.index:
                start_ts = sliced_prices.index[0]
                feat_from_start = feat.loc[start_ts:]
                if horizon_steps > 0:
                    sliced_futures[name] = feat_from_start.iloc[
                        : len(sliced_prices) + horizon_steps
                    ]
                else:
                    sliced_futures[name] = feat_from_start
            elif end_date in feat.index:
                loc = feat.index.get_loc(end_date)
                assert isinstance(loc, int)
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

        sliced_ra = self.risk_aversion
        if isinstance(self.risk_aversion, pd.Series):
            sliced_ra = self.risk_aversion.loc[:end_date]
            if lookback is not None:
                sliced_ra = sliced_ra.iloc[-lookback:]

        return self.__class__(
            assets=list(self.assets),
            prices=sliced_prices,
            features=sliced_features,
            future_features=sliced_futures,
            cross_sectional=self.cross_sectional,
            risk_free_rate=sliced_rf,
            risk_aversion=sliced_ra,
        )

    def get_returns(self, log_returns: bool = False) -> pd.DataFrame:
        returns = self.prices.pct_change()
        if log_returns:
            returns = cast(pd.DataFrame, np.log1p(returns))
        return returns.dropna()

    def extract_covariates(
        self,
        mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        horizon: int | None = None,
        dtype: Any = np.float32,
        return_channel_names: bool = False,
    ) -> Any:
        """
        Standardized, optional exogenous covariate extractor shared across `ApolloPFN`,
        `TimesFM-3`, and `Timestral`.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None] or tuple[np.ndarray | None, np.ndarray | None, list[str]]
            - `past_covariates`: shape `(C_past, T)` or `None` when no past covariates match
            - `future_covariates`: shape `(C_future, T + H)` or `None` when no future covariates match
            - `channel_names` (if `return_channel_names=True`): list of covariate channel labels
        """
        if mode == CovariateMode.NONE:
            if return_channel_names:
                return None, None, []
            return None, None

        assets_set = set(self.assets)
        selected_set = (
            {str(k.value if isinstance(k, CovariateKey) else k) for k in covariate_keys}
            if covariate_keys is not None
            else None
        )
        t_steps = len(self.prices)
        channel_names: list[str] = []

        def _should_include(key: str, df: pd.DataFrame) -> tuple[bool, bool]:
            str_key = str(key.value if isinstance(key, CovariateKey) else key)
            if selected_set is not None and str_key not in selected_set:
                return False, False
            if mode == CovariateMode.SELECTED and selected_set is None:
                return False, False
            cols = [str(c) for c in df.columns]
            is_asset_aligned = len(cols) > 0 and set(cols).issuperset(assets_set)
            if mode == CovariateMode.ASSET_ONLY and not is_asset_aligned:
                return False, False
            if mode == CovariateMode.MACRO_ONLY and is_asset_aligned:
                return False, False
            return True, is_asset_aligned

        past_list: list[np.ndarray] = []
        if self.features:
            for feat_key, df_feat in self.features.items():
                inc, is_asset = _should_include(feat_key, df_feat)
                if not inc:
                    continue
                k_label = str(
                    feat_key.value if isinstance(feat_key, CovariateKey) else feat_key
                )
                if is_asset:
                    aligned = (
                        df_feat.reindex(index=self.prices.index, columns=self.assets)
                        .ffill()
                        .fillna(0.0)
                    )
                    for col in self.assets:
                        channel_names.append(f"{k_label}:{col}")
                else:
                    aligned = (
                        df_feat.reindex(index=self.prices.index).ffill().fillna(0.0)
                    )
                    for col in aligned.columns:
                        channel_names.append(f"{k_label}:{col}")
                past_list.append(
                    np.nan_to_num(
                        aligned.to_numpy(dtype=dtype).T, nan=0.0, posinf=0.0, neginf=0.0
                    )
                )

        past_cov = np.concatenate(past_list, axis=0) if past_list else None

        future_list: list[np.ndarray] = []
        if self.future_features:
            for feat_key, df_fut in self.future_features.items():
                inc, is_asset = _should_include(feat_key, df_fut)
                if not inc:
                    continue
                k_label = str(
                    feat_key.value if isinstance(feat_key, CovariateKey) else feat_key
                )
                df_fut_sliced = df_fut
                if (
                    len(self.prices.index) > 0
                    and len(df_fut.index) > 0
                    and df_fut.index[0] < self.prices.index[0]
                ):
                    df_fut_sliced = df_fut.loc[self.prices.index[0] :]

                if is_asset:
                    aligned_f = (
                        df_fut_sliced.reindex(columns=self.assets).ffill().fillna(0.0)
                    )
                    for col in self.assets:
                        channel_names.append(f"future_{k_label}:{col}")
                else:
                    aligned_f = df_fut_sliced.ffill().fillna(0.0)
                    for col in aligned_f.columns:
                        channel_names.append(f"future_{k_label}:{col}")
                arr = np.nan_to_num(
                    aligned_f.to_numpy(dtype=dtype).T, nan=0.0, posinf=0.0, neginf=0.0
                )
                if horizon is not None:
                    expected_len = t_steps + int(horizon)
                    if arr.shape[1] < expected_len:
                        pad_w = expected_len - arr.shape[1]
                        arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="edge")
                    elif arr.shape[1] > expected_len:
                        arr = arr[:, :expected_len]
                future_list.append(arr)

        future_cov = np.concatenate(future_list, axis=0) if future_list else None

        if return_channel_names:
            return past_cov, future_cov, channel_names
        return past_cov, future_cov


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
    initial_prices: np.ndarray | None = None
    timestamps: pd.Index = field(default_factory=lambda: pd.RangeIndex(0))
    features: dict[str, np.ndarray] = field(default_factory=dict)
    feature_columns: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_paths, self.horizon, self.n_assets = self.prices.shape
        if len(self.timestamps) == 0:
            self.timestamps = pd.RangeIndex(start=0, stop=self.horizon, step=1)

    def get_cumulative_returns(self) -> np.ndarray:
        """
        Returns total cumulative asset returns of shape (M, N) over the full horizon,
        incorporating t=0 initial_prices when available.
        """
        final_prices = self.prices[:, -1, :]
        if self.initial_prices is not None:
            init_p = np.asarray(self.initial_prices, dtype=final_prices.dtype)
            return (final_prices - init_p) / init_p
        initial_prices = self.prices[:, 0, :]
        return (final_prices - initial_prices) / initial_prices

    def get_path(
        self, path_idx: int, feature_pipeline: object | None = None
    ) -> MarketData:
        """
        Slices the 3D tensor at index `path_idx` and returns a standard 2D MarketData
        object mathematically compatible with the existing StrategyPipeline.
        """
        path_prices = pd.DataFrame(
            self.prices[path_idx], index=self.timestamps, columns=self.asset_names
        )

        path_features: dict[str, pd.DataFrame] = {}
        for feature_name, feature_tensor in self.features.items():
            # Support for both asset-specific features (M, T, N) and exogenous macro features (M, T, K)
            if feature_tensor.ndim == 3 and feature_tensor.shape[0] == self.n_paths:
                if feature_name in self.feature_columns:
                    columns: list[str] | None = self.feature_columns[feature_name]
                elif feature_tensor.shape[2] == self.n_assets:
                    columns = self.asset_names
                else:
                    columns = None
                path_features[feature_name] = pd.DataFrame(
                    feature_tensor[path_idx], index=self.timestamps, columns=columns
                )

        market_data = MarketData(prices=path_prices, features=path_features)
        if feature_pipeline is not None and hasattr(feature_pipeline, "run"):
            market_data = feature_pipeline.run(market_data)
        return market_data
