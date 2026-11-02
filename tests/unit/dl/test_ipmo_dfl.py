import numpy as np
import pandas as pd
import pytest
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.results import MultiPeriodOptimizationResult

from pyalloq.dl.estimators.e2e.ipmo_allocator import (
    IPMOAllocator,
    IPMOModule,
    MultiHorizonPredictorNet,
)
from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer
from pyalloq.dl.loss_functions.dfl import HybridDFLLoss, SPOPlusLoss

pytest.importorskip("torch")


def test_mdfp_layer_forward_simplex_and_implicit_backward() -> None:
    torch.manual_seed(0)
    B, H, N = 3, 4, 5
    mu_seq = (torch.randn(B, H, N, dtype=torch.float32) * 0.01).requires_grad_(True)
    eye = torch.eye(N, dtype=torch.float32) * 0.01
    cov_seq = eye.view(1, 1, N, N).expand(B, H, N, N).clone().requires_grad_(True)
    w_curr = torch.full((B, N), 1.0 / N, dtype=torch.float32, requires_grad=True)

    layer = MDFPOptimizationLayer(
        horizon=H,
        risk_aversion=2.0,
        cost_penalty=0.001,
        max_iter=40,
        implicit_backward=True,
    )
    z_star = layer(mu_seq, cov_seq=cov_seq, w_current=w_curr)

    assert z_star.shape == (B, H, N)
    # Every stage weight vector lies on the probability simplex
    sums = z_star.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert bool((z_star >= 0.0).all())

    # Verify implicit fixed-point differentiation produces finite, non-zero gradients
    target = torch.zeros_like(z_star)
    target[:, :, 0] = 1.0
    loss = ((z_star - target) ** 2).mean()
    loss.backward()

    assert mu_seq.grad is not None
    assert cov_seq.grad is not None
    assert w_curr.grad is not None
    assert bool(torch.isfinite(mu_seq.grad).all())
    assert float(mu_seq.grad.abs().sum().item()) > 0.0


def test_spo_plus_and_hybrid_dfl_losses() -> None:
    torch.manual_seed(11)
    B, H, N = 4, 3, 4
    mu_hat = (torch.randn(B, H, N) * 0.01).requires_grad_(True)
    y_true = torch.randn(B, H, N) * 0.01

    spo_loss_fn = SPOPlusLoss(horizon=H, risk_aversion=2.0, cost_penalty=0.001)
    loss_spo = spo_loss_fn(mu_hat, y_true)
    assert loss_spo.ndim == 0
    assert float(loss_spo.item()) >= 0.0
    loss_spo.backward()
    assert mu_hat.grad is not None
    assert bool(torch.isfinite(mu_hat.grad).all())

    mu_hat.grad = None
    mdfp = MDFPOptimizationLayer(horizon=H)
    z_traj = mdfp(mu_hat)
    hybrid_fn = HybridDFLLoss(alpha=0.6, cost_penalty=0.001)
    loss_hyb = hybrid_fn(z_traj, mu_hat, y_true)
    loss_hyb.backward()
    assert mu_hat.grad is not None
    assert bool(torch.isfinite(mu_hat.grad).all())


def test_ipmo_allocator_and_module_end_to_end() -> None:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2023-01-02", periods=50)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (50, 4)), axis=0)),
        index=dates,
        columns=tickers,
    )
    mom = prices.pct_change().rolling(5, min_periods=1).mean().fillna(0.0)
    md = MarketData(prices=prices, features={"mom": mom})

    predictor = MultiHorizonPredictorNet(n_assets=4, n_features=2, horizon=4)
    mdfp = MDFPOptimizationLayer(horizon=4)
    ipmo_mod = IPMOModule(predictor=predictor, mdfp_layer=mdfp)

    dummy_x = torch.randn(2, 15, 4, 2)
    w_immediate = ipmo_mod(dummy_x, return_trajectory=False)
    assert isinstance(w_immediate, torch.Tensor)
    assert w_immediate.shape == (2, 4)

    allocator = IPMOAllocator(
        predictor=predictor,
        mdfp_layer=mdfp,
        lookback_window=15,
    )
    current_w = pd.Series([0.4, 0.3, 0.2, 0.1], index=tickers)
    res = allocator.allocate(md, current_weights=current_w)

    assert isinstance(res, MultiPeriodOptimizationResult)
    assert res.status == "OPTIMAL_IPMO_MDFP"
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
    assert res.weights_trajectory.shape == (4, 4)
    assert res.trades_trajectory.shape == (4, 4)
