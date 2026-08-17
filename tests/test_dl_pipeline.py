import numpy as np
import pandas as pd
import torch

from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.enums import ObjectiveFunction
from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_backtest.costs import FlatBpsCostModel

from pyalloq.deep_learning.loss_functions.sharpe import SharpeLoss
from pyalloq.deep_learning.estimators.returns.models.tft import TFTDeepReturnsNet

from pyalloq.estimators.covariance.ewma import EWMACovariance
from pyalloq.deep_learning.estimators.returns.deep_return import DeepReturnEstimator
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


def test_full_pipeline() -> None:
    print("1. Generating Market Data...")
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    data = generate_synthetic_market_data(assets=assets)
    n_features = 3  # returns + volume + rsi
    lookback = 30

    print("2. Initializing Deep Learning Models...")
    tft_model = TFTDeepReturnsNet(n_features=n_features, hidden_size=32, num_heads=2)

    # 3. Setup Splitter and Trainer
    splitter = RollingWindowSplitter(lookback_window=120)
    rebalance_dates = pd.DatetimeIndex(data.prices.resample("ME").last().index)

    print("3. Training TFT Return Estimator...")
    tft_trainer = WalkForwardTrainer(
        model=tft_model,
        loss_function=SharpeLoss(risk_free_rate=0.03),
        optimizer=torch.optim.Adam(tft_model.parameters(), lr=1e-3),
        splitter=splitter,
        lookback_window=lookback,
        epochs_per_window=2,
    )
    tft_trainer.train(data, rebalance_dates)

    print("4. Assembling Hybrid Strategy Pipeline...")
    # Wrap PyTorch models into BaseEstimator wrappers
    returns_est = DeepReturnEstimator(model=tft_model, lookback_window=lookback)
    cov_est = EWMACovariance(span=30)

    pipeline = StrategyPipeline(
        returns_estimator=returns_est,
        cov_estimator=cov_est,
        allocator=MarkowitzAllocator(
            tickers=assets, objective=ObjectiveFunction.MAX_RETURN
        ),
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
    print("Backtest Execution Completed Successfully!")
    print(f"Tear Sheet: {results['tear_sheet']}")


if __name__ == "__main__":
    test_full_pipeline()
