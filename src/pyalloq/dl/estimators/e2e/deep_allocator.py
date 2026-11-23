from typing import Any

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from torch import nn


class DeepAllocator(BaseAllocator):
    """
    End-to-End PyTorch DL Allocator mapping raw features directly to portfolio weights.
    Supports ragged tradability masking (``mask``) and optional cash sleeve outputs ``(B, N+1)``.
    """

    def __init__(
        self,
        model: nn.Module,
        lookback_window: int = 60,
        device: str = "cpu",
        tickers: list[str] | None = None,
        graph_prior: Any | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.lookback_window = lookback_window
        self.graph_prior = graph_prior
        self.last_cash_weight_: float = 0.0
        self.last_attn_bias_: np.ndarray | None = None

    def _resolve_attn_bias(
        self,
        data: MarketData,
        x_batch: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        explicit_bias = kwargs.get("attn_bias")
        if explicit_bias is not None:
            if isinstance(explicit_bias, torch.Tensor):
                return explicit_bias.to(self.device)
            return torch.tensor(
                np.asarray(explicit_bias, dtype=np.float32), device=self.device
            )

        prior_src = kwargs.get("graph_prior", self.graph_prior)
        if prior_src is None:
            if data.features and any(
                k in data.features
                for k in ("relational_graph", "news_graph", "entity_graph")
            ):
                from pyalloq.dl.layers.relational_graph import (
                    RelationalGraphPriorModule,
                )

                mod = RelationalGraphPriorModule().to(self.device)
                bias_t = mod.compute_attn_bias(data, x_batch)
                if bias_t is not None:
                    self.last_attn_bias_ = bias_t.squeeze(0).cpu().numpy()
                return bias_t
            return None

        if hasattr(prior_src, "compute_attn_bias"):
            bias_t = prior_src.compute_attn_bias(data, x_batch)
            if isinstance(bias_t, torch.Tensor):
                self.last_attn_bias_ = bias_t.squeeze(0).cpu().numpy()
                return bias_t.to(self.device)
        if isinstance(prior_src, pd.DataFrame):
            assets = list(data.prices.columns)
            arr = prior_src.reindex(
                index=assets, columns=assets, fill_value=0.0
            ).to_numpy(dtype=np.float32)
            self.last_attn_bias_ = arr
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        if isinstance(prior_src, np.ndarray):
            arr = np.asarray(prior_src, dtype=np.float32)
            self.last_attn_bias_ = arr
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        if isinstance(prior_src, torch.Tensor):
            self.last_attn_bias_ = prior_src.detach().cpu().numpy()
            return prior_src.to(self.device)
        return None

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

        raw_prices_win = data.prices.iloc[-self.lookback_window :]
        raw_rets_win = data.prices.pct_change(fill_method=None).iloc[
            -self.lookback_window :
        ]
        # Asset is tradable at decision time iff its price and return window are finite
        active_mask_np = (
            raw_prices_win.notna().all(axis=0).to_numpy(dtype=bool)
            & np.isfinite(raw_prices_win.to_numpy(dtype=np.float64)).all(axis=0)
            & raw_rets_win.notna().all(axis=0).to_numpy(dtype=bool)
        )
        if not active_mask_np.any():
            active_mask_np = np.ones(n_assets, dtype=bool)
        mask_batch = torch.tensor(
            active_mask_np, dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        returns_df = raw_rets_win.fillna(0.0)
        base_tensor = torch.tensor(returns_df.values, dtype=torch.float32).unsqueeze(-1)
        tensor_list = [base_tensor]

        if data.features is not None:
            for df_feat in data.features.values():
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

        attn_bias = self._resolve_attn_bias(data, x_batch, kwargs)
        with torch.no_grad():
            try:
                if w_curr_tensor is not None:
                    if attn_bias is not None:
                        weights_tensor = self.model(
                            x_batch,
                            w_current=w_curr_tensor,
                            mask=mask_batch,
                            attn_bias=attn_bias,
                        )
                    else:
                        weights_tensor = self.model(
                            x_batch, w_current=w_curr_tensor, mask=mask_batch
                        )
                else:
                    if attn_bias is not None:
                        weights_tensor = self.model(
                            x_batch, mask=mask_batch, attn_bias=attn_bias
                        )
                    else:
                        weights_tensor = self.model(x_batch, mask=mask_batch)
            except TypeError:
                try:
                    if w_curr_tensor is not None:
                        weights_tensor = self.model(x_batch, w_current=w_curr_tensor)
                    else:
                        weights_tensor = self.model(x_batch)
                except TypeError:
                    weights_tensor = self.model(x_batch)

        weights_np = weights_tensor.squeeze(0).cpu().numpy()
        if len(weights_np) == n_assets + 1:
            self.last_cash_weight_ = float(weights_np[-1])
            weights_np = weights_np[:n_assets]
        else:
            self.last_cash_weight_ = 0.0

        weights = pd.Series(weights_np, index=data.prices.columns, name="weights")

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_DL_E2E",
            metadata={"cash_weight": self.last_cash_weight_},
        )
