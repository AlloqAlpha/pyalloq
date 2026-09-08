import pandas as pd
import numpy as np
from typing import Any
from timesfm3 import TimesFM3Evaluator
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseViewGenerator

P10_INDEX = 0
P50_INDEX = 4
P90_INDEX = 8


class TimesFM3UnivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (Q) and uncertainty (Omega)
    using the probabilistic quantile outputs of TimesFM-3.0
    """

    def __init__(
        self,
        tfm_model: TimesFM3Evaluator,
        horizon: int = 21,
    ) -> None:
        self.tfm_model = tfm_model
        self.horizon = horizon


def generate(
    self,
    data: MarketData,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    prices_df = data.prices

    # FIX 1: Explicitly cast columns to a list to avoid Mypy ExtensionArray errors
    from typing import cast

    assets = list(cast(Any, prices_df.columns))
    N = len(assets)

    # FIX 2: Convert to numpy FIRST, then slice the last day to get current prices
    current_prices = prices_df.to_numpy(dtype=np.float32)[-1]

    # FIX 3: Build the Multivariate Target Tensor.
    # TimesFM expects shape (N_assets, Context_Length).
    # We simply convert the (Time, N_assets) DataFrame and Transpose it (.T)
    target = prices_df.to_numpy(dtype=np.float32).T

    # Optional: Dynamically load features (past-only) and future_features if they exist
    past_only_cov: np.ndarray | None = None
    if data.features:
        past_covs: list[np.ndarray] = []
        for feat_name, raw_feat in data.features.items():
            df_feat = cast(pd.DataFrame, raw_feat)
            aligned_feat = df_feat.reindex(
                index=prices_df.index, columns=assets
            ).fillna(0.0)
            past_covs.append(aligned_feat.to_numpy(dtype=np.float32).T)
        past_only_cov = np.concatenate(past_covs, axis=0)

    past_future_cov: np.ndarray | None = None
    if getattr(data, "future_features", None):
        future_covs: list[np.ndarray] = []
        for feat_name, raw_feat in data.future_features.items():
            df_feat = cast(pd.DataFrame, raw_feat)
            aligned_feat = df_feat.reindex(columns=assets).fillna(0.0)
            future_covs.append(aligned_feat.to_numpy(dtype=np.float32).T)
        past_future_cov = np.concatenate(future_covs, axis=0)

    # 4. Build predict_batch kwargs dynamically
    predict_kwargs: dict[str, Any] = {
        "contexts": [target],
        "horizon": self.horizon,
        "return_quantiles": True,
        "use_symmetric_averaging": False,
    }
    if past_only_cov is not None:
        predict_kwargs["past_only_covariates"] = [past_only_cov]
    if past_future_cov is not None:
        predict_kwargs["past_future_covariates"] = [past_future_cov]

    # 5. Predict and extract
    # Cast to list[Any] to keep Mypy from complaining about dynamic API attributes
    outputs = cast(list[Any], list(self.tfm_model.predict_batch(**predict_kwargs)))

    # For multivariate, outputs[0].quantiles shape is (N_Assets, Horizon, 9)
    # Index 0 = p10, Index 4 = p50 (median), Index 8 = p90
    quantiles = outputs[0].quantiles
    p10_prices = quantiles[:, self.horizon - 1, 0]
    p50_prices = quantiles[:, self.horizon - 1, 4]
    p90_prices = quantiles[:, self.horizon - 1, 8]

    # 6. View Expected Returns (Q) - based on the median forecast
    view_returns = (p50_prices / current_prices) - 1.0
    annualized_returns = view_returns * (252.0 / self.horizon)
    Q = pd.Series(annualized_returns, index=assets)

    # 7. View Uncertainty Matrix (Omega)
    # We estimate standard deviation from the quantile spread: std ≈ (p90 - p10) / 2.56
    price_std = (p90_prices - p10_prices) / 2.56
    return_std = price_std / current_prices
    annualized_std = return_std * np.sqrt(252.0 / self.horizon)

    # Omega is a diagonal matrix of these variances
    omega_variances = cast(np.ndarray, annualized_std**2)
    Omega = pd.DataFrame(np.diag(omega_variances), index=assets, columns=assets)

    # 8. Picking Matrix (P) is the Identity Matrix since we forecast every asset
    P = pd.DataFrame(np.eye(N), index=assets, columns=assets)

    return P, Q, Omega


class TimesFMM3ultivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (Q) and uncertainty (Omega)
    using the probabilistic outputs of TimesFM 3.0's Multivariate setup.
    """

    def __init__(self, model: TimesFM3Evaluator, horizon: int = 21):
        self.model = model
        self.horizon = horizon

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        stock_prices = data.prices
        assets = data.assets
        N = len(assets)
        current_prices = stock_prices.to_numpy(dtype=np.float32)[-1]

        # 1. Target Variates: TimesFM expects shape (num_variates, context_length)
        # We transpose our (T, N) price DataFrame to (N, T)
        target = stock_prices.to_numpy(dtype=np.float32)

        # 2. Past-Only Covariates: Process MarketData.features
        # TimesFM expects shape (num_covariate_channels, context_length)
        past_covs = []
        past_only_cov: np.ndarray | None = None
        if data.features is not None:
            for feat_name, df_feat in data.features.items():
                # Reindex to ensure strict alignment with prices, just in case
                aligned_feature = df_feat.reindex(
                    index=stock_prices.index, columns=assets
                ).fillna(0.0)

                # Transpose to (N, T). Each asset's feature becomes a distinct covariate channel
                past_covs.append(aligned_feature.to_numpy(dtype=np.float32).T)

            past_only_cov = np.concatenate(past_covs, axis=0)
        else:
            past_only_cov = None

        predict_kwargs = {
            "contexts": [target],
            "horizon": self.horizon,
            "return_quantiles": True,
            "use_symmetric_averaging": False,
        }
        if past_only_cov is not None:
            predict_kwargs["past_only_covariates"] = [past_only_cov]

        # 4. Generate the Joint Multivariate Forecast
        outputs = self.model.predict_batch(**predict_kwargs)
        outputs = list(outputs)

        # 5. Extract Quantiles for the specific horizon
        # For multivariate, outputs[0].quantiles shape is (N, Horizon, 9)
        quantiles = outputs[0].quantiles
        p10_prices = quantiles[:, self.horizon - 1, P10_INDEX]
        p50_prices = quantiles[:, self.horizon - 1, P50_INDEX]
        p90_prices = quantiles[:, self.horizon - 1, P90_INDEX]

        # 6. View Expected Returns (Q)
        view_returns = (p50_prices / current_prices) - 1.0
        annualized_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(annualized_returns, index=assets)

        # 7. View Uncertainty Matrix (Omega)
        # Standard deviation estimated from the spread: std ≈ (p90 - p10) / 2.56
        price_std = (p90_prices - p10_prices) / 2.56
        return_std = price_std / current_prices
        annualized_std = return_std * np.sqrt(252.0 / self.horizon)

        # Omega is a diagonal matrix of these variances
        Omega = pd.DataFrame(np.diag(annualized_std**2), index=assets, columns=assets)

        # 8. Picking Matrix (P)
        P = pd.DataFrame(np.eye(N), index=assets, columns=assets)

        return P, Q, Omega
