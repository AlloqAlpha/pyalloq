import pandas as pd
import numpy as np
from typing import Any

try:
    from timesfm3 import TimesFM3Evaluator
except ImportError:
    TimesFM3Evaluator = Any  # type: ignore[misc,assignment]

from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseViewGenerator

P10_INDEX = 0
P50_INDEX = 4
P90_INDEX = 8


class TimesFM3UnivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (Q) and uncertainty (Omega)
    using the probabilistic quantile outputs of TimesFM-3.0 in univariate mode.
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
        from typing import cast

        assets = list(cast(Any, prices_df.columns))
        N = len(assets)
        current_prices = prices_df.to_numpy(dtype=np.float32)[-1]

        # 1. TimesFM expects a list of 1D numpy arrays for univariate series
        ts_list = [prices_df[col].to_numpy(dtype=np.float32) for col in assets]

        # 2. Run zero-shot probabilistic forecasting
        outputs = cast(
            list[Any],
            list(
                self.tfm_model.predict_batch(
                    ts_list,
                    horizon=self.horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            ),
        )

        # 3. Extract quantiles for each asset (index 0 = p10, index 4 = p50, index 8 = p90)
        p10_prices = np.array(
            [out.quantiles[self.horizon - 1, P10_INDEX] for out in outputs]
        )
        p50_prices = np.array(
            [out.quantiles[self.horizon - 1, P50_INDEX] for out in outputs]
        )
        p90_prices = np.array(
            [out.quantiles[self.horizon - 1, P90_INDEX] for out in outputs]
        )

        # 4. View Expected Returns (Q) - based on the median forecast
        view_returns = (p50_prices / current_prices) - 1.0
        annualized_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(annualized_returns, index=assets, name="expected_returns")

        # 5. View Uncertainty Matrix (Omega)
        price_std = (p90_prices - p10_prices) / 2.56
        return_std = price_std / current_prices
        annualized_std = return_std * np.sqrt(252.0 / self.horizon)

        omega_variances = cast(np.ndarray, annualized_std**2)
        Omega = pd.DataFrame(np.diag(omega_variances), index=assets, columns=assets)

        # 6. Picking Matrix (P) is the Identity Matrix
        P = pd.DataFrame(np.eye(N), index=assets, columns=assets)

        return P, Q, Omega


class TimesFM3MultivariateViewGenerator(BaseViewGenerator):
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
        target = stock_prices.to_numpy(dtype=np.float32).T

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
