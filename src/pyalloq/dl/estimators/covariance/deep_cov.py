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
        graph_prior: Any | None = None,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.lookback_window = lookback_window
        self.allow_ragged = allow_ragged
        self.graph_prior = graph_prior
        self.inactive_diagonal_penalty = 1e6
        self.last_factors_: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.last_tradability_mask_: pd.Series | None = None
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
            # Check if data.features carries a relational_graph / news_graph matrix
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
        attn_bias = self._resolve_attn_bias(data, x_batch, {})
        with torch.no_grad():
            if attn_bias is not None:
                try:
                    b_load, sigma_f, d_diag = forward_factors_fn(
                        x_batch, mask=mask_batch, attn_bias=attn_bias
                    )
                except TypeError:
                    b_load, sigma_f, d_diag = forward_factors_fn(
                        x_batch, mask=mask_batch
                    )
            else:
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

        idx = len(dataset) - 1
        item = dataset[idx]
        x_tensor = item[0]
        x_batch = x_tensor.unsqueeze(0).to(self.device)
        mask_tensor = dataset.get_tradability_mask(idx)
        mask_np = mask_tensor.cpu().numpy().astype(bool)
        self.last_tradability_mask_ = pd.Series(
            mask_np, index=data.prices.columns, name="tradability_mask"
        )

        attn_bias = self._resolve_attn_bias(data, x_batch, kwargs)
        with torch.no_grad():
            if self.allow_ragged or len(item) == 3:
                mask_batch = mask_tensor.unsqueeze(0).to(self.device)
                try:
                    if attn_bias is not None:
                        cov_pred_tensor = self.model(
                            x_batch, mask=mask_batch, attn_bias=attn_bias
                        )
                    else:
                        cov_pred_tensor = self.model(x_batch, mask=mask_batch)
                except TypeError:
                    cov_pred_tensor = self.model(x_batch)
            else:
                try:
                    if attn_bias is not None:
                        cov_pred_tensor = self.model(x_batch, attn_bias=attn_bias)
                    else:
                        cov_pred_tensor = self.model(x_batch)
                except TypeError:
                    cov_pred_tensor = self.model(x_batch)
        cov_np = cov_pred_tensor.squeeze(0).cpu().numpy().astype(np.float64)

        if self.allow_ragged and not np.all(mask_np):
            # Ensure masked (pre-IPO / delisted) slots have zero cross-covariance with active assets
            # and a large positive diagonal penalty so the full N x N matrix remains strictly SPD
            # for CVXPY DCP while classical variance/risk-parity solvers assign 0 weight to masked slots
            inactive_idx = np.flatnonzero(~mask_np)
            cov_np[inactive_idx, :] = 0.0
            cov_np[:, inactive_idx] = 0.0
            cov_np[inactive_idx, inactive_idx] = self.inactive_diagonal_penalty

        cov_matrix = pd.DataFrame(
            cov_np,
            index=data.prices.columns,
            columns=data.prices.columns,
        )
        cov_matrix.attrs["tradability_mask"] = mask_np.tolist()

        return cov_matrix
