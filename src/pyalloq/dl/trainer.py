from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import torch
from pyalloq_backtest.splitters import BaseWindowSplitter
from pyalloq_core.data import MarketData
from torch import nn
from torch.utils.data import DataLoader

from pyalloq.dl.dataset import MarketDataset, UniverseSubsampler


class WalkForwardTrainer:
    """
    Trains a PyTorch Model using Walk-Forward optimization.
    Ingests any PyAlloq Window Splitter to ensure the model trains exactly
    how it will be backtested, preventing look-ahead bias.

    Supports optional ragged tradability masking (``return_mask=True``),
    Multi-Cardinality Subsampling (``universe_subsampler``), and ``O(N m^2 + m^3)``
    low-rank factor forwarding (``use_forward_factors=True``) for Woodbury losses.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        splitter: BaseWindowSplitter,
        lookback_window: int = 60,
        horizon: int = 1,
        batch_size: int = 32,
        epochs_per_window: int = 10,
        device: str = "cpu",
        return_mask: bool = False,
        allow_ragged: bool = False,
        universe_subsampler: UniverseSubsampler | None = None,
        use_forward_factors: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_function.to(device)
        self.optimizer = optimizer
        self.splitter = splitter

        self.lookback_window = lookback_window
        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs_per_window = epochs_per_window
        self.device = device
        self.return_mask = bool(return_mask or universe_subsampler is not None)
        self.allow_ragged = bool(allow_ragged or self.return_mask)
        self.universe_subsampler = universe_subsampler
        self.use_forward_factors = bool(use_forward_factors)

        self.initial_state_dict = copy.deepcopy(model.state_dict())

    def _forward_model(
        self,
        batch_X: torch.Tensor,
        batch_mask: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_forward_factors and hasattr(self.model, "forward_factors"):
            model_any: Any = self.model
            if batch_mask is not None:
                return model_any.forward_factors(batch_X, mask=batch_mask)
            return model_any.forward_factors(batch_X)

        if batch_mask is not None:
            try:
                return self.model(batch_X, mask=batch_mask)
            except TypeError:
                return self.model(batch_X)
        return self.model(batch_X)

    def _compute_loss(
        self,
        predictions: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_y: torch.Tensor,
        batch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if batch_mask is not None:
            try:
                return self.loss_fn(predictions, batch_y, mask=batch_mask)
            except TypeError:
                if isinstance(predictions, torch.Tensor):
                    # Mask inactive return targets before unmasked loss evaluation
                    m_f = batch_mask.to(device=batch_y.device, dtype=batch_y.dtype)
                    if batch_y.dim() == 3:
                        y_m = batch_y * m_f.unsqueeze(1)
                    else:
                        y_m = batch_y * m_f
                    return self.loss_fn(predictions, y_m)
        return self.loss_fn(predictions, batch_y)

    def train(
        self,
        data: MarketData,
        rebalance_dates: pd.DatetimeIndex,
        reset_weights_each_window: bool = False,
    ) -> dict[str, list[float]]:
        history: dict = {
            "window_date": [],
            "final_loss": [],
        }

        for current_date, train_window_data in self.splitter.split(
            data, rebalance_dates
        ):
            print(f"--- Training up to {current_date.date()} ---")
            try:
                dataset = MarketDataset(
                    data=train_window_data,
                    lookback_window=self.lookback_window,
                    horizon=self.horizon,
                    return_mask=self.return_mask,
                    allow_ragged=self.allow_ragged,
                )
            except ValueError as e:
                print(f"Skipping {current_date.date()}: {e}")
                continue

            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            if reset_weights_each_window:
                self.model.load_state_dict(copy.deepcopy(self.initial_state_dict))

            self.model.train()
            final_epoch_loss = 0.0
            for epoch in range(self.epochs_per_window):
                epoch_loss = 0.0

                for batch in dataloader:
                    if len(batch) == 3:
                        batch_X, batch_y, batch_mask = batch
                        batch_mask = batch_mask.to(self.device)
                    else:
                        batch_X, batch_y = batch
                        batch_mask = None

                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if self.universe_subsampler is not None:
                        batch_X, batch_mask = self.universe_subsampler(
                            batch_X, batch_mask
                        )

                    self.optimizer.zero_grad()
                    predictions = self._forward_model(batch_X, batch_mask)
                    loss = self._compute_loss(predictions, batch_y, batch_mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                    epoch_loss += loss.item()
                mean_loss = epoch_loss / len(dataloader)
                final_epoch_loss = mean_loss

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(
                        f"  Epoch [{epoch + 1}/{self.epochs_per_window}], Loss: {mean_loss:.6f}"
                    )

            history["window_date"].append(current_date)
            history["final_loss"].append(final_epoch_loss)

        return history
