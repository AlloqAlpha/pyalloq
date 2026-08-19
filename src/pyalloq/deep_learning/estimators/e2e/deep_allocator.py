import torch
import torch.nn as nn
import pandas as pd
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

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        returns_df = data.prices.pct_change().fillna(0.0).iloc[-self.lookback_window :]
        base_tensor = torch.tensor(returns_df.values, dtype=torch.float32).unsqueeze(-1)
        tensor_list = [base_tensor]

        for feat_name, df_feat in data.features.items():
            feat_window = df_feat.iloc[-self.lookback_window :]
            feat_tensor = torch.tensor(
                feat_window.values, dtype=torch.float32
            ).unsqueeze(-1)
            tensor_list.append(feat_tensor)

        x_tensor = torch.cat(tensor_list, dim=-1)

        x_batch = x_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            weights_tensor = self.model(x_batch)

        weights_np = weights_tensor.squeeze(0).cpu().numpy()
        weights = pd.Series(weights_np, index=data.prices.columns, name="weights")

        return OptimizationResult(weights=weights, status="OPTIMAL_DL_E2E")
