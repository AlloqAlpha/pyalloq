import torch
import torch.nn as nn
import pandas as pd
from typing import Any

from pyalloq_core.interfaces import BaseCovarianceEstimator
from pyalloq_core.data import MarketData
from pyalloq.deep_learning.dataset import MarketDataset


class DeepCovarianceEstimator(BaseCovarianceEstimator):
    """
    Wraps a trained Deep Learning model to forecast covariance matrix for downstream optimizers.
    """

    def __init__(
        self,
        model: nn.Module,
        lookback_window: int = 60,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.lookback_window = lookback_window

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if len(data.prices) < self.lookback_window:
            cov_np = data.prices.pct_change().cov().fillna(0.0).values
            return pd.DataFrame(
                cov_np, index=data.prices.columns, columns=data.prices.columns
            )

        dataset = MarketDataset(
            data=data,
            lookback_window=self.lookback_window,
            horizon=1,
        )

        x_tensor, _ = dataset[len(dataset) - 1]
        x_batch = x_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Output Shape: (1, N_Assets, N_Assets)
            cov_pred_tensor = self.model(x_batch)
        cov_np = cov_pred_tensor.squeeze(0).cpu().numpy()

        cov_matrix = pd.DataFrame(
            cov_np,
            index=data.prices.columns,
            columns=data.prices.columns,
        )

        return cov_matrix
