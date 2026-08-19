import numpy as np
import pandas as pd
import torch

from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_backtest.costs import FlatBpsCostModel

# --- Deep Learning E2E Imports ---
from pyalloq.deep_learning.loss_functions.sharpe import SharpeLoss
from pyalloq.deep_learning.estimators.e2e.models.c_stan import (
    ConstrainedSpatioTemporalAttentionNet,
)
from pyalloq.deep_learning.estimators.e2e.deep_allocator import DeepAllocator
from pyalloq.deep_learning.trainer import WalkForwardTrainer


def generate_synthetic_market_data(assets: list[str], n_days: int = 500) -> MarketData:
    """Generates synthetic price and feature data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="B")

    # Generate geometric Brownian motion prices
    returns = np.random.normal(0.0005, 0.015, size=(n_days, len(assets)))
    price_paths = 100.0 * np.cumprod(1 + returns, axis=0)
    df_prices = pd.DataFrame(price_paths, index=dates, columns=assets)

    # Generate synthetic features
    volume = np.random.uniform(1e6, 5e6, size=(n_days, len(assets)))
    df_volume = pd.DataFrame(volume, index=dates, columns=assets)

    rsi = np.random.uniform(20.0, 80.0, size=(n_days, len(assets)))
    df_rsi = pd.DataFrame(rsi, index=dates, columns=assets)

    return MarketData(
        prices=df_prices,
        features={"volume": df_volume, "rsi": df_rsi},
    )


def test_e2e_pipeline() -> None:
    print("1. Generating Market Data...")
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    data = generate_synthetic_market_data(assets=assets)
    n_features = 3  # returns (calculated internally) + volume + rsi
    lookback = 30

    print("2. Initializing E2E Deep Allocation Network (CSTAN)...")
    # Initialize our new Constrained Spatio-Temporal Attention Network
    cstan_model = ConstrainedSpatioTemporalAttentionNet(
        n_features=n_features,
        hidden_dim=32,
        num_heads=2,
        max_asset_weight=0.35,  # Strict max cap per asset
        waterfill_iterations=3,
    )

    # 3. Setup Splitter and Trainer
    splitter = RollingWindowSplitter(lookback_window=120)
    rebalance_dates = pd.DatetimeIndex(data.prices.resample("ME").last().index)

    print("3. Training CSTAN E2E Allocator (Sharpe Maximization)...")
    cstan_trainer = WalkForwardTrainer(
        model=cstan_model,
        loss_function=SharpeLoss(
            risk_free_rate=0.03
        ),  # Cost-aware Sharpe Loss natively
        optimizer=torch.optim.Adam(cstan_model.parameters(), lr=1e-3),
        splitter=splitter,
        lookback_window=lookback,
        epochs_per_window=2,
    )
    # The trainer logic is identical—it just learns weights now instead of expected returns
    cstan_trainer.train(data, rebalance_dates)

    print("4. Assembling E2E Strategy Pipeline...")
    # Notice: We completely drop ReturnsEstimator, CovarianceEstimator, and Markowitz.
    # The DeepAllocator wraps the CSTAN model and does everything natively.
    e2e_allocator = DeepAllocator(
        model=cstan_model, lookback_window=lookback, device="cpu"
    )

    pipeline = StrategyPipeline(
        returns_estimator=None,  # Bypassed
        cov_estimator=None,  # Bypassed
        allocator=e2e_allocator,
    )

    print("5. Running Walk-Forward Backtest...")
    engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=splitter,
        cost_model=FlatBpsCostModel(),
        rebalance_freq="ME",
    )

    results = engine.run(data)

    print("\n================ SUCCESS ================")
    print("E2E DL Backtest Execution Completed Successfully!")
    print(f"Tear Sheet: {results['tear_sheet']}")


if __name__ == "__main__":
    test_e2e_pipeline()
