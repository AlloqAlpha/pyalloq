import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import copy

from pyalloq_core.data import MarketData
from pyalloq_backtest.splitters import BaseWindowSplitter
from pyalloq.deep_learning.dataset import MarketDataset


class WalkForwardTrainer:
    """
    Trains a Pytorch Model using Walk Forward optimization.
    Ingest any pyalloq Window Splitter to ensure the model trains exactly
    how it will be backtested, preventing look-ahead bias.
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

        self.initial_state_dict = copy.deepcopy(model.state_dict())

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

                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self.optimizer.zero_grad()
                    predictions = self.model(batch_X)
                    loss = self.loss_fn(predictions, batch_y)
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
                        f"  Epoch [{epoch+1}/{self.epochs_per_window}], Loss: {mean_loss:.6f}"
                    )

            history["window_date"].append(current_date)
            history["final_loss"].append(final_epoch_loss)

        return history
