import pytest
import torch
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from torch import nn

from pyalloq.dl.trainer import WalkForwardTrainer


class SimplePredictor(nn.Module):
    def __init__(self, n_assets: int, n_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(n_assets * n_features, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Lookback, N_assets, N_features)
        # Use the last time step
        last_step = x[:, -1, :, :].reshape(x.size(0), -1)
        out = self.fc(last_step)
        return out.unsqueeze(1)  # (Batch, 1, N_assets)


@pytest.mark.dl
class TestWalkForwardTrainer:
    def test_walk_forward_training(self, synthetic_market_data: MarketData) -> None:
        n_assets = len(synthetic_market_data.assets)
        n_features = 1 + len(synthetic_market_data.features or {})
        model = SimplePredictor(n_assets, n_features)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        splitter = RollingWindowSplitter(lookback_window=60)
        rebalance_dates = synthetic_market_data.prices.index[60::20]

        trainer = WalkForwardTrainer(
            model=model,
            loss_function=loss_fn,
            optimizer=optimizer,
            splitter=splitter,
            lookback_window=20,
            horizon=1,
            batch_size=8,
            epochs_per_window=2,
            device="cpu",
        )

        history = trainer.train(
            synthetic_market_data,
            rebalance_dates=rebalance_dates,
            reset_weights_each_window=True,
        )

        assert "window_date" in history
        assert "final_loss" in history
        assert len(history["window_date"]) > 0
        assert len(history["final_loss"]) == len(history["window_date"])
        assert all(isinstance(loss, float) for loss in history["final_loss"])
