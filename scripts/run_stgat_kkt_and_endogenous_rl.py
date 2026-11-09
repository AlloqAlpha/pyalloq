"""
End-to-End Spatio-Temporal Graph (CRISP ST-GAT) + Differentiable KKT Classical Allocation
& Endogenous Liquidity Impact RL Example:
1. Builds PIT Rough Volatility & Log-Signature features on `MarketData`.
2. Instantiates `CRISPSparseSTGAT` (combining `NeuralJumpRDE`, CRISP sparse graph attention,
   and dynamic `FragilityShrinkageCholeskyLayer`) chained into:
   - `DifferentiableRiskParityLayer` (exact closed-form KKT Jacobian)
   - `DifferentiableCVaRLayer` (implicit fixed-point differentiation)
   via `DifferentiableKKTAllocatorModule`.
3. Trains the end-to-end ST-GAT + KKT Risk Parity network with `WalkForwardTrainer` and
   `SharpeLoss`, then backtests it via `DeepAllocator` inside `WalkForwardEngine`.
4. Demonstrates `EndogenousLiquidityImpactWrapper` wrapping `PortfolioGymEnv` for reactive
   order-book resilience and Graph Laplacian contagion cost accounting.
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import torch
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_features import (
    LiftedHestonRoughVolTransformer,
    LogSignatureTransformer,
    PITOptionsMFIVTransformer,
)

from pyalloq import SpectralLaplacianHRPAllocator
from pyalloq.dl import (
    CRISPSparseSTGAT,
    DeepCovarianceEstimator,
    DifferentiableKKTAllocatorModule,
)
from pyalloq.dl.estimators.e2e.deep_allocator import DeepAllocator
from pyalloq.dl.loss_functions.sharpe import SharpeLoss
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.rl import EndogenousLiquidityImpactWrapper, PortfolioGymEnv

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]


def build_featured_market_data() -> MarketData:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(dt.datetime(2023, 1, 2), dt.datetime(2024, 3, 29))
    n_days, n_assets = len(dates), len(TICKERS)

    mkt = rng.normal(0.0004, 0.011, size=n_days)
    rets = np.column_stack(
        [0.75 * mkt + rng.normal(0.0001, 0.009, size=n_days) for _ in range(n_assets)]
    )
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=TICKERS,
    )
    data = MarketData(prices=prices, risk_free_rate=0.03)
    data = PITOptionsMFIVTransformer().transform(data)
    data = LiftedHestonRoughVolTransformer(n_factors=4, estimation_window=20).transform(
        data
    )
    data = LogSignatureTransformer(window=6, depth=2).transform(data)
    return data


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    market_data = build_featured_market_data()
    n_assets = len(market_data.assets)
    n_features = 1 + len(market_data.features)
    lookback_window = 20

    print(
        "--- 1. Training End-to-End CRISPSparseSTGAT + DifferentiableRiskParityLayer ---"
    )
    stgat_backbone = CRISPSparseSTGAT(
        num_assets=n_assets,
        num_features=n_features,
        hidden_dim=16,
        num_heads=4,
        keep_ratio=0.4,
    )
    e2e_kkt_model = DifferentiableKKTAllocatorModule(
        backbone=stgat_backbone,
        mode="risk_parity",
    )

    cost_model = FlatBpsCostModel(bps=10.0)
    loss_fn = SharpeLoss(cost_model=cost_model, risk_free_rate=0.03 / 252.0)
    optimizer = torch.optim.AdamW(
        e2e_kkt_model.parameters(), lr=2e-3, weight_decay=1e-4
    )

    splitter = RollingWindowSplitter(lookback_window=100)
    idx_ser = pd.Series(market_data.prices.index, index=market_data.prices.index)
    rebalance_dates = pd.DatetimeIndex(
        idx_ser.resample("QE").last().dropna().values
    ).unique()

    trainer = WalkForwardTrainer(
        model=e2e_kkt_model,
        loss_function=loss_fn,
        optimizer=optimizer,
        splitter=splitter,
        lookback_window=lookback_window,
        horizon=1,
        batch_size=16,
        epochs_per_window=3,
    )
    trainer.train(data=market_data, rebalance_dates=rebalance_dates)

    print(
        "\n--- 2. Backtesting Trained ST-GAT KKT Allocator vs. ST-GAT + Spectral HRP ---"
    )
    kkt_deep_allocator = DeepAllocator(
        model=e2e_kkt_model,
        lookback_window=lookback_window,
        tickers=TICKERS,
    )
    pipe_kkt = StrategyPipeline(allocator=kkt_deep_allocator)

    pipe_stgat_hrp = StrategyPipeline(
        cov_estimator=DeepCovarianceEstimator(
            model=stgat_backbone, lookback_window=lookback_window
        ),
        allocator=SpectralLaplacianHRPAllocator(
            tickers=TICKERS, fragility_penalty=0.25
        ),
    )

    bt_kkt = WalkForwardEngine(
        pipeline=pipe_kkt, splitter=splitter, cost_model=cost_model, rebalance_freq="ME"
    ).run(market_data)
    bt_stgat_hrp = WalkForwardEngine(
        pipeline=pipe_stgat_hrp,
        splitter=splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
    ).run(market_data)

    comparison = pd.DataFrame(
        {
            "STGAT_DiffKKT_RiskParity": bt_kkt.tear_sheet["Value"],
            "STGAT_SpectralLaplacian_HRP": bt_stgat_hrp.tear_sheet["Value"],
        }
    )
    print(comparison.round(4))

    print("\n--- 3. Stepping PortfolioGymEnv with EndogenousLiquidityImpactWrapper ---")
    env = EndogenousLiquidityImpactWrapper(
        PortfolioGymEnv(market_data, lookback_window=15),
        impact_lambda=0.008,
        resilience_rho=0.65,
        diffusion_kappa=0.003,
    )
    obs, info = env.reset(seed=42)
    total_impact = 0.0
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(n_assets,)).astype(np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_impact += float(step_info["endogenous_impact_cost"])
        if terminated or truncated:
            break
    print(
        f"Completed 10 RL steps | Cumulative Endogenous Impact Cost: {total_impact:.6f} | "
        f"Terminal Portfolio Value: {env.portfolio_value:.4f}"
    )


if __name__ == "__main__":
    main()
