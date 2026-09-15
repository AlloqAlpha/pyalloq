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
