from abc import ABC, abstractmethod
from collections.abc import Iterator
import numpy as np
import pandas as pd
from typing import Any
from .enums import DataFrequency
from .data import MarketData, ScenarioMarketData
from .results import OptimizationResult


class BaseAllocator(ABC):
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def _resolve_annualization_factor(
        self,
        returns: pd.DataFrame,
        frequency: DataFrequency | str | int | None = None,
    ) -> float:
        """Determines annualization factor from Enum, int or inferred pandas index."""
        if isinstance(frequency, DataFrequency):
            return float(frequency.value)
        if isinstance(frequency, (int, float)):
            return float(frequency)
        if isinstance(frequency, str):
            try:
                float(DataFrequency[frequency.upper()].value)
            except KeyError:
                pass

        if isinstance(returns.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(returns.index)
            if inferred:
                if "W" in inferred:
                    return DataFrequency.WEEKLY.value
                if "M" in inferred or "MS" in inferred:
                    return DataFrequency.MONTHLY.value
                if "B" in inferred or "D" in inferred:
                    return DataFrequency.DAILY.value

        return DataFrequency.DAILY.value

    def _prepare_inputs(
        self,
        prices: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        window_len: int | None = None,
        frequency: DataFrequency | str | int = DataFrequency.DAILY,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Internal helper to resolve inputs. If raw prices are given,
        it computes basic historical matrices. Otherwise, it uses the provided ones.
        """
        final_returns = expected_returns
        final_cov = cov_matrix

        returns = prices.iloc[-window_len:] if window_len else prices
        scale_factor = self._resolve_annualization_factor(returns, frequency)
        df_returns = returns.pct_change().dropna()

        if final_returns is None:
            final_returns = df_returns.mean() * scale_factor

        if final_cov is None:
            final_cov = df_returns.cov() * scale_factor

        if final_returns is None or final_cov is None:
            raise ValueError(
                "You must provide either raw 'prices' or BOTH 'expected_returns' and 'cov_matrix'"
            )

        return (final_returns, final_cov)

    @abstractmethod
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Base allocate function all subclasses must implement
        """
        ...


class BaseViewGenerator(ABC):
    @abstractmethod
    def generate(
        self, data: MarketData, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Generates P, Q and Omega that will be used for Black Litterman
        """
        ...


class BaseReturnEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        """
        Takes raw prices (and optional macro/technical features) and
        returns an Nx1 pandas Series of expected returns for the assets.
        """
        pass


class BaseCovarianceEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        """Takes a price DataFrame and returns an NxN covariance matrix."""
        pass


class BaseScenarioGenerator(ABC):
    dtype: np.dtype[Any] = np.dtype(np.float32)
    _feature_meta: dict[str, dict[str, Any]]
    _feature_return_mean: np.ndarray | None = None
    _last_timestamp: pd.Timestamp | None = None
    _inferred_freq: str = "B"
    _is_datetime_index: bool = False

    @abstractmethod
    def fit(self, data: MarketData) -> None:
        """
        Fits the underlying statistical or generative model.
        """
        ...

    @abstractmethod
    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        """
        Outputs a 3D simulation tensor.
        """
        ...

    def generate_batches(
        self, n_paths: int, horizon: int, batch_size: int = 1000
    ) -> Iterator[ScenarioMarketData]:
        """
        Yields simulated ScenarioMarketData in memory-bounded batches.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        remaining = n_paths
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            yield self.generate(n_paths=current_batch, horizon=horizon)
            remaining -= current_batch

    def _record_index_metadata(self, index: pd.Index) -> None:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
            self._is_datetime_index = True
            self._last_timestamp = pd.Timestamp(index[-1])
            inferred = pd.infer_freq(index) if len(index) >= 3 else None
            self._inferred_freq = inferred if inferred is not None else "B"
        else:
            self._is_datetime_index = False
            self._last_timestamp = None

    def _build_future_timestamps(self, horizon: int) -> pd.Index:
        if (
            getattr(self, "_is_datetime_index", False)
            and self._last_timestamp is not None
        ):
            freq = getattr(self, "_inferred_freq", "B") or "B"
            try:
                future_idx = pd.date_range(
                    start=self._last_timestamp, periods=horizon + 1, freq=freq
                )[1:]
                return future_idx
            except Exception:
                return pd.bdate_range(start=self._last_timestamp, periods=horizon + 1)[
                    1:
                ]
        return pd.RangeIndex(start=0, stop=horizon, step=1)

    def _fit_exogenous_features(self, data: MarketData, returns: pd.DataFrame) -> None:
        """
        Fits a conditional joint projection model mapping asset returns to concurrent
        changes in exogenous features (data.features) so that simulated futures
        preserve cross-correlations between asset returns and features.
        """
        self._feature_meta = {}
        if not data.features:
            return

        R = returns.to_numpy(dtype=np.float64)
        T_r, N = R.shape
        r_mean = np.mean(R, axis=0)
        R_centered = R - r_mean
        self._feature_return_mean = r_mean

        # Ridge-stabilized precision for OLS projection B = (R^T R + reg I)^(-1) R^T dX
        gram = R_centered.T @ R_centered + 1e-6 * np.eye(N)

        for feat_name, df_feat in data.features.items():
            clean_df = df_feat.ffill().bfill().fillna(0.0)
            diff_df = clean_df.diff().reindex(returns.index).fillna(0.0)
            dX = diff_df.to_numpy(dtype=np.float64)
            dx_mean = np.mean(dX, axis=0)
            dX_centered = dX - dx_mean

            B = np.linalg.solve(gram, R_centered.T @ dX_centered)  # (N, K_f)
            residuals = dX_centered - (R_centered @ B)
            resid_std = np.std(residuals, axis=0)

            self._feature_meta[feat_name] = {
                "columns": [str(c) for c in clean_df.columns],
                "initial": clean_df.iloc[-1].to_numpy(dtype=np.float64),
                "dx_mean": dx_mean,
                "beta": B,
                "resid_std": resid_std,
                "min_val": np.min(clean_df.to_numpy(dtype=np.float64), axis=0),
                "strictly_positive": bool(
                    np.all(clean_df.to_numpy(dtype=np.float64) > 0)
                ),
            }

    def _simulate_exogenous_features(
        self, synthetic_returns: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """
        Synthesizes 3D feature tensors (M, T, K_f) conditioned on synthetic_returns (M, T, N).
        """
        if (
            not getattr(self, "_feature_meta", None)
            or self._feature_return_mean is None
        ):
            return {}, {}

        n_paths, horizon, _ = synthetic_returns.shape
        out_dtype = synthetic_returns.dtype
        R_centered = synthetic_returns - self._feature_return_mean.astype(out_dtype)

        features_out: dict[str, np.ndarray] = {}
        columns_out: dict[str, list[str]] = {}

        for feat_name, meta in self._feature_meta.items():
            k_f = len(meta["columns"])
            beta = meta["beta"].astype(out_dtype)
            dx_mean = meta["dx_mean"].astype(out_dtype)
            resid_std = meta["resid_std"].astype(out_dtype)
            initial = meta["initial"].astype(out_dtype)

            noise = np.random.standard_normal(size=(n_paths, horizon, k_f)).astype(
                out_dtype
            )
            dx_sim = dx_mean + np.matmul(R_centered, beta) + noise * resid_std
            feat_paths = initial + np.cumsum(dx_sim, axis=1)

            if meta["strictly_positive"]:
                floor = np.maximum(meta["min_val"].astype(out_dtype) * 0.1, 1e-6)
                feat_paths = np.maximum(feat_paths, floor)

            features_out[feat_name] = feat_paths.astype(out_dtype, copy=False)
            columns_out[feat_name] = meta["columns"]

        return features_out, columns_out
