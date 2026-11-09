from typing import Any

import pandas as pd
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from torch import nn


class DeepAllocator(BaseAllocator):
    """
    End-to-End Pytorch DL Allocator mapping raw features directly to portfolio weights.
    """

    def __init__(
        self,
        model: nn.Module,
        lookback_window: int = 60,
        device: str = "cpu",
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
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
        if not self.tickers and data.assets:
            self.tickers = list(data.assets)
        tickers = list(data.prices.columns)
        n_assets = len(tickers)
        returns_df = data.prices.pct_change().fillna(0.0).iloc[-self.lookback_window :]
        base_tensor = torch.tensor(returns_df.values, dtype=torch.float32).unsqueeze(-1)
        tensor_list = [base_tensor]

        if data.features is not None:
            for feat_name, df_feat in data.features.items():
                if (
                    df_feat.shape[1] == 1
                    and n_assets > 1
                    and df_feat.columns[0] not in tickers
                ):
                    col_ser = (
                        df_feat.iloc[:, 0]
                        .reindex(returns_df.index)
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                    )
                    feat_np = col_ser.to_numpy(dtype="float32")[:, None].repeat(
                        n_assets, axis=1
                    )
                else:
                    feat_window = (
                        df_feat.reindex(index=returns_df.index, columns=tickers)
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                    )
                    feat_np = feat_window.to_numpy(dtype="float32")
                feat_tensor = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(-1)
                tensor_list.append(feat_tensor)

        x_tensor = torch.cat(tensor_list, dim=-1)

        x_batch = x_tensor.unsqueeze(0).to(self.device)

        current_weights = kwargs.get("current_weights")
        w_curr_tensor: torch.Tensor | None = None
        if current_weights is not None:
            if isinstance(current_weights, pd.Series):
                w_np = (
                    current_weights.reindex(tickers)
                    .fillna(0.0)
                    .to_numpy(dtype="float32")
                )
            else:
                w_np = pd.Series(current_weights, index=tickers).to_numpy(
                    dtype="float32"
                )
            w_curr_tensor = torch.tensor(
                w_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        with torch.no_grad():
            try:
                if w_curr_tensor is not None:
                    weights_tensor = self.model(x_batch, w_current=w_curr_tensor)
                else:
                    weights_tensor = self.model(x_batch)
            except TypeError:
                weights_tensor = self.model(x_batch)

        weights_np = weights_tensor.squeeze(0).cpu().numpy()
        weights = pd.Series(weights_np, index=data.prices.columns, name="weights")

        return OptimizationResult(weights=weights, status="OPTIMAL_DL_E2E")
