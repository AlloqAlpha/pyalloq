import numpy as np
import pandas as pd
import pytest
import torch
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_core.data import MarketData

from pyalloq.dl.costs import TorchCostAdapter
from pyalloq.dl.dataset import MarketDataset


@pytest.mark.dl
class TestMarketDataset:
    def test_dataset_without_features(self, synthetic_market_data: MarketData) -> None:
        # Create MarketData without additional features
        clean_data = MarketData(
            prices=synthetic_market_data.prices,
            risk_free_rate=0.02,
        )
        lookback = 30
        horizon = 5
        dataset = MarketDataset(clean_data, lookback_window=lookback, horizon=horizon)

        n_returns = len(clean_data.prices) - 1
        expected_len = n_returns - lookback - horizon + 1
        assert len(dataset) == expected_len

        x, y = dataset[0]
        # X: (Lookback, N_assets, 1 feature)
        assert x.shape == (lookback, len(clean_data.assets), 1)
        # y: (Horizon, N_assets)
        assert y.shape == (horizon, len(clean_data.assets))
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_dataset_with_features(self, synthetic_market_data: MarketData) -> None:
        lookback = 40
        horizon = 1
        dataset = MarketDataset(
            synthetic_market_data, lookback_window=lookback, horizon=horizon
        )

        n_features = 1 + len(synthetic_market_data.features or {})
        x, y = dataset[0]
        assert x.shape == (lookback, len(synthetic_market_data.assets), n_features)
        # Horizon 1 is unsqueezed to (1, N_assets)
        assert y.shape == (1, len(synthetic_market_data.assets))

    def test_dataset_too_short_raises_value_error(self) -> None:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"A": np.linspace(10, 15, 10)}, index=dates)
        short_data = MarketData(prices=prices)

        with pytest.raises(ValueError, match="too short"):
            MarketDataset(short_data, lookback_window=20, horizon=1)


@pytest.mark.dl
class TestTorchCostAdapter:
    def test_flat_cost_adapter(self) -> None:
        cost_model = FlatBpsCostModel(bps=10.0)
        adapter = TorchCostAdapter(cost_model=cost_model)

        weights_delta = torch.tensor([[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32)
        costs = adapter(weights_delta)

        # Expected: abs(delta) * (10 / 10000)
        expected = torch.abs(weights_delta) * 0.001
        assert torch.allclose(costs, expected)

    def test_cost_adapter_with_volume(self) -> None:
        cost_model = FlatBpsCostModel(bps=5.0)
        adapter = TorchCostAdapter(cost_model=cost_model)

        weights_delta = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
        volume = torch.tensor([[10000.0, 20000.0]], dtype=torch.float32)

        costs = adapter(weights_delta, volume=volume)
        assert costs.shape == weights_delta.shape
