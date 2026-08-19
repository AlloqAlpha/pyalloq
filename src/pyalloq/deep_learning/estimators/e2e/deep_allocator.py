import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Any
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.data import MarketData
from pyalloq_core.results import OptimizationResult


class DeepAllocator(BaseAllocator):
    """
    End-to-End Pytorch DL Allocator mapping raw features directly to portfolio weights.
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

    def allocator(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> OptimizationResult:
        prices = data.prices
        if len(prices) < self.lookback_window:
            n_assets = len(prices.columns)
            eq_weights = pd.Series(1.0 / n_assets, index=prices.columns)
            return OptimizationResult(weights=eq_weights, status="WARMUP_FALLBACK")

        window_data = prices.iloc[-self.lookback_window :].to_numpy()
        x_batch = (
            torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            weights_tensor = self.model(x_batch)
        weights_np = weights_tensor.squeeze(0).cpu().numpy()

        if np.isnan(weights_np).any():
            raise ValueError(
                "E2E Model output NaN. Check input data or training gradients."
            )

        weights_np = weights_np / weights_np.sum()
        weights = pd.Series(weights_np, index=prices.columns, name="weights")

        return OptimizationResult(weights=weights, status="OPTIMAL_DL_E2E")
