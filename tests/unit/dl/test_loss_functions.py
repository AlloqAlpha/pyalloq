import pytest
import torch

from pyalloq.dl.loss_functions.cholesky_nll import CholeskyNLLLoss
from pyalloq.dl.loss_functions.sharpe import SharpeLoss


@pytest.mark.dl
class TestLossFunctions:
    def test_sharpe_loss_2d(self) -> None:
        loss_fn = SharpeLoss(risk_free_rate=0.0)
        weights = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.3, 0.7]], requires_grad=True)
        returns = torch.tensor([[0.02, 0.01], [0.03, -0.01], [0.01, 0.04]])

        loss = loss_fn(weights, returns)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

        loss.backward()
        assert weights.grad is not None
        assert not torch.isnan(weights.grad).any()

    def test_sharpe_loss_2d_weights_with_3d_dataset_returns(self) -> None:
        """
        Regression test: MarketDataset yields y of shape (Batch, 1, N_assets) while
        E2E models (like C-STAN) output weights of shape (Batch, N_assets).
        Verify SharpeLoss aligns (B, 1, N) without cross-broadcasting (B, B, N).
        """
        from pyalloq_backtest.costs import FlatBpsCostModel

        loss_fn = SharpeLoss(cost_model=FlatBpsCostModel(bps=0.0), risk_free_rate=0.0)
        weights = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]], dtype=torch.float32
        )
        # 3D target from MarketDataset with horizon=1: shape (4, 1, 2)
        returns_3d = torch.tensor(
            [[[0.01, -0.05]], [[-0.02, 0.03]], [[0.02, 0.02]], [[0.00, 0.01]]],
            dtype=torch.float32,
        )

        # Expected per-sample portfolio returns: [0.01, 0.03, 0.02, 0.008]
        expected_port_rets = torch.tensor(
            [0.01, 0.03, 0.02, 0.008], dtype=torch.float32
        )
        expected_sharpe = expected_port_rets.mean() / (
            expected_port_rets.std() + loss_fn.eps
        )

        loss = loss_fn(weights, returns_3d)
        assert torch.isclose(loss, -expected_sharpe, atol=1e-5)

    def test_sharpe_loss_3d_with_volume_and_initial_weights(self) -> None:
        loss_fn = SharpeLoss()
        batch_size = 4
        time_steps = 3
        n_assets = 2

        weights = torch.full(
            (batch_size, time_steps, n_assets), 0.5, requires_grad=True
        )
        returns = torch.normal(0.001, 0.02, size=(batch_size, time_steps, n_assets))
        volume = torch.full((batch_size, time_steps, n_assets), 1000.0)
        initial_weights = torch.zeros((batch_size, n_assets))

        loss = loss_fn(
            weights,
            returns,
            volume_matrix=volume,
            initial_weights=initial_weights,
        )
        assert loss.numel() == 1
        assert not torch.isnan(loss)

        loss.backward()
        assert weights.grad is not None
        assert not torch.isnan(weights.grad).any()

    def test_cholesky_nll_loss(self) -> None:
        loss_fn = CholeskyNLLLoss()
        batch_size = 5
        n_assets = 3

        # Construct valid lower-triangular L with positive diagonals
        raw = torch.randn(batch_size, n_assets, n_assets, requires_grad=True)
        L = torch.tril(raw, diagonal=-1) + torch.diag_embed(
            torch.exp(torch.diagonal(raw, dim1=-2, dim2=-1))
        )

        y_true = torch.randn(batch_size, n_assets)

        loss = loss_fn(L, y_true)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

        loss.backward()
        assert raw.grad is not None
        assert not torch.isnan(raw.grad).any()
