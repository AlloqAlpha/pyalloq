import torch
import torch.nn as nn
import pandas as pd
from typing import Any

from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.data import MarketData
from pyalloq.deep_learning.dataset import MarketDataset


class DeepReturnEstimator(BaseReturnEstimator):
    """
    Wraps a trained Deep Learning model to forecast expected returns for downstream optimizers
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
    ) -> pd.Series:
        if len(data.prices) < self.lookback_window:
            return pd.Series(0.0, index=data.prices.columns, name="expected_returns")

        dataset = MarketDataset(
            data=data,
            lookback_window=self.lookback_window,
            horizon=1,
        )

        x_tensor, _ = dataset[len(dataset) - 1]
        x_batch = x_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu_pred_tensor = self.model(x_batch)
        mu_pred = mu_pred_tensor.squeeze(0).cpu().numpy()

        return pd.Series(
            mu_pred,
            index=data.prices.columns,
            name="expected_returns",
        )
