from typing import Any

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseCovarianceEstimator
from torch import nn

from pyalloq.dl.dataset import MarketDataset


class DeepCovarianceEstimator(BaseCovarianceEstimator):
    """
    Wraps a trained Deep Learning model to forecast covariance matrix for downstream optimizers.
    Supports both dense ``forward(x) -> (B, N, N)`` networks and mask-aware / low-rank
    ``SetTransformerCovarianceNet`` models.
    """

    def __init__(
        self,
        model: nn.Module,
        lookback_window: int = 60,
        device: str = "cpu",
        allow_ragged: bool = False,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.lookback_window = lookback_window
        self.allow_ragged = allow_ragged
        self.last_factors_: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _prepare_latest_batch(
        self, data: MarketData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = MarketDataset(
            data=data,
            lookback_window=self.lookback_window,
            horizon=1,
            return_mask=True,
            allow_ragged=self.allow_ragged,
        )
        idx = len(dataset) - 1
        item = dataset[idx]
        x_tensor = item[0]
        mask_tensor = dataset.get_tradability_mask(idx)
        x_batch = x_tensor.unsqueeze(0).to(self.device)
        mask_batch = mask_tensor.unsqueeze(0).to(self.device)
        return x_batch, mask_batch

    def estimate_factors(
        self,
        data: MarketData,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Extracts the low-rank factor decomposition ``(B_load_df, Sigma_F_np, d_diag_series)``
        when ``self.model`` implements ``forward_factors(x, mask=...)``.
        """
        forward_factors_fn = getattr(self.model, "forward_factors", None)
        if not callable(forward_factors_fn):
            raise TypeError(
                f"Model {type(self.model).__name__} does not implement `forward_factors`."
            )
        x_batch, mask_batch = self._prepare_latest_batch(data)
        with torch.no_grad():
            b_load, sigma_f, d_diag = forward_factors_fn(x_batch, mask=mask_batch)
        b_np = b_load.squeeze(0).cpu().numpy()
        sf_np = sigma_f.squeeze(0).cpu().numpy()
        d_np = d_diag.squeeze(0).cpu().numpy()
        self.last_factors_ = (b_np, sf_np, d_np)

        tickers = list(data.prices.columns)
        factor_cols = [f"Factor_{k}" for k in range(b_np.shape[1])]
        return (
            pd.DataFrame(b_np, index=tickers, columns=factor_cols),
            sf_np,
            pd.Series(d_np, index=tickers, name="idiosyncratic_variance"),
        )

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if len(data.prices) < self.lookback_window + 2:
            cov_np = data.prices.pct_change().cov().fillna(0.0).values
            return pd.DataFrame(
                cov_np, index=data.prices.columns, columns=data.prices.columns
            )

        dataset = MarketDataset(
            data=data,
            lookback_window=self.lookback_window,
            horizon=1,
            return_mask=self.allow_ragged,
            allow_ragged=self.allow_ragged,
        )

        item = dataset[len(dataset) - 1]
        x_tensor = item[0]
        x_batch = x_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if len(item) == 3:
                mask_batch = item[2].unsqueeze(0).to(self.device)
                try:
                    cov_pred_tensor = self.model(x_batch, mask=mask_batch)
                except TypeError:
                    cov_pred_tensor = self.model(x_batch)
            else:
                cov_pred_tensor = self.model(x_batch)
        cov_np = cov_pred_tensor.squeeze(0).cpu().numpy()

        cov_matrix = pd.DataFrame(
            cov_np,
            index=data.prices.columns,
            columns=data.prices.columns,
        )

        return cov_matrix
