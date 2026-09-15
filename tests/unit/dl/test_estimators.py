import numpy as np
import pandas as pd
import pytest
import torch
from pyalloq_core.data import MarketData
from torch import nn

from pyalloq.dl.estimators.covariance.deep_cov import DeepCovarianceEstimator
from pyalloq.dl.estimators.covariance.models.cac import CrossAttentionCholeskyNet
from pyalloq.dl.estimators.e2e.deep_allocator import DeepAllocator
from pyalloq.dl.estimators.e2e.models.c_stan import (
    BoundedSoftmax,
    ConstrainedSpatioTemporalAttentionNet,
)
from pyalloq.dl.estimators.returns.deep_return import DeepReturnEstimator


class DummyReturnModel(nn.Module):
    def __init__(self, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Lookback, N_assets, N_features)
        batch_size = x.size(0)
        return torch.full((batch_size, self.n_assets), 0.05)


class DummyCovModel(nn.Module):
    def __init__(self, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        eye = torch.eye(self.n_assets).unsqueeze(0).repeat(batch_size, 1, 1)
        return eye * 0.04


class DummyAllocModel(nn.Module):
    def __init__(self, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.full((batch_size, self.n_assets), 1.0 / self.n_assets)


@pytest.mark.dl
class TestDeepEstimators:
    def test_deep_return_estimator(self, synthetic_market_data: MarketData) -> None:
        n_assets = len(synthetic_market_data.assets)
        model = DummyReturnModel(n_assets)
        estimator = DeepReturnEstimator(model, lookback_window=30)

        # 1. Normal estimation
        mu = estimator.estimate(synthetic_market_data)
        assert isinstance(mu, pd.Series)
        assert list(mu.index) == synthetic_market_data.assets
        assert np.allclose(mu.values, 0.05)

        # 2. Fallback when data is too short
        short_data = MarketData(prices=synthetic_market_data.prices.iloc[:10])
        fallback_mu = estimator.estimate(short_data)
        assert isinstance(fallback_mu, pd.Series)
        assert np.allclose(fallback_mu.values, 0.0)

    def test_deep_cov_estimator(self, synthetic_market_data: MarketData) -> None:
        n_assets = len(synthetic_market_data.assets)
        model = DummyCovModel(n_assets)
        estimator = DeepCovarianceEstimator(model, lookback_window=30)

        # 1. Normal estimation
        cov = estimator.estimate(synthetic_market_data)
        assert isinstance(cov, pd.DataFrame)
        assert list(cov.index) == synthetic_market_data.assets
        assert list(cov.columns) == synthetic_market_data.assets
        assert np.allclose(cov.values, np.eye(n_assets) * 0.04)

        # 2. Fallback on short data
        short_data = MarketData(prices=synthetic_market_data.prices.iloc[:10])
        fallback_cov = estimator.estimate(short_data)
        assert isinstance(fallback_cov, pd.DataFrame)
        assert fallback_cov.shape == (n_assets, n_assets)

    def test_deep_allocator(self, synthetic_market_data: MarketData) -> None:
        n_assets = len(synthetic_market_data.assets)
        model = DummyAllocModel(n_assets)
        allocator = DeepAllocator(model, lookback_window=30)

        # Test with features
        result = allocator.allocate(synthetic_market_data)
        assert result.status == "OPTIMAL_DL_E2E"
        assert isinstance(result.weights, pd.Series)
        assert np.isclose(result.weights.sum(), 1.0)

        # Test without features
        no_feat_data = MarketData(prices=synthetic_market_data.prices)
        result_no_feat = allocator.allocate(no_feat_data)
        assert result_no_feat.status == "OPTIMAL_DL_E2E"
        assert np.isclose(result_no_feat.weights.sum(), 1.0)


@pytest.mark.dl
class TestNeuralArchitectures:
    def test_cross_attention_cholesky_net(self) -> None:
        batch_size = 2
        lookback = 20
        n_assets = 4
        n_features = 3

        net = CrossAttentionCholeskyNet(
            n_features=n_features, hidden_dim=16, num_heads=2
        )
        x = torch.randn(batch_size, lookback, n_assets, n_features)
        cov = net(x)

        assert cov.shape == (batch_size, n_assets, n_assets)
        # Check positive definiteness (eigenvalues > 0)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues > 0).all()

    def test_bounded_softmax(self) -> None:
        bounded = BoundedSoftmax(max_weight=0.35, iterations=5)
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.2]])
        weights = bounded(logits)

        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert (weights <= 0.35 + 1e-5).all()

    def test_cstan_allocator_net(self) -> None:
        batch_size = 2
        lookback = 15
        n_assets = 5
        n_features = 2

        net = ConstrainedSpatioTemporalAttentionNet(
            n_features=n_features,
            hidden_dim=16,
            num_heads=2,
            max_asset_weight=0.30,
        )
        x = torch.randn(batch_size, lookback, n_assets, n_features)
        weights = net(x)

        assert weights.shape == (batch_size, n_assets)
        # Verify sum to 1 and capped
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        assert (weights <= 0.30 + 1e-4).all()
