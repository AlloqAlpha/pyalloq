import torch
import torch.nn as nn
import pandas as pd
from typing import Any
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.data import MarketData
from pyalloq.core.results import OptimizationResult
from pyalloq.deep_learning.dataset import MarketDataset


class DeepAllocator(BaseAllocator):
    """
    Wraps a trained Pytorch model into a standard pyalloq Allocator.
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

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        dataset = MarketDataset(
            data=data, lookback_window=self.lookback_window, horizon=1
        )

        x_tensor, _ = dataset[-1]

        x_batch = x_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            weights_tensor = self.model(x_batch)

        weights = weights_tensor.squeeze(0).cpu().numpy()
        weights = pd.Series(weights, index=data.prices.columns, name="weights")

        return OptimizationResult(weights=weights, status="OPTIMAL_DL_E2E")
