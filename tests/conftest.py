from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData


@pytest.fixture
def sample_assets() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL", "AMZN"]


@pytest.fixture
def synthetic_prices(sample_assets: list[str]) -> pd.DataFrame:
    """Generates 252 business days of synthetic daily prices for sample assets."""
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.015, size=(n_days, len(sample_assets)))
    price_paths = 100.0 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(price_paths, index=dates, columns=sample_assets)


@pytest.fixture
def synthetic_market_data(
    synthetic_prices: pd.DataFrame, sample_assets: list[str]
) -> MarketData:
    """Creates a standard MarketData instance with prices, features, and risk-free rate."""
    np.random.seed(42)
    n_days = len(synthetic_prices)
    dates = synthetic_prices.index

    # Features aligned with prices index
    volume = pd.DataFrame(
        np.random.uniform(1e6, 5e6, size=(n_days, len(sample_assets))),
        index=dates,
        columns=sample_assets,
    )
    rsi = pd.DataFrame(
        np.random.uniform(20.0, 80.0, size=(n_days, len(sample_assets))),
        index=dates,
        columns=sample_assets,
    )

    return MarketData(
        prices=synthetic_prices,
        features={"volume": volume, "rsi": rsi},
        risk_free_rate=0.03,
        risk_aversion=1.0,
    )


@pytest.fixture
def collinear_market_data() -> MarketData:
    """Creates MarketData with collinear assets to test rank deficiency and ill-conditioned covariance."""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    base_returns = np.random.normal(0.0005, 0.01, size=n_days)

    price_a = 100.0 * np.cumprod(1 + base_returns)
    price_b = price_a * 2.0  # Perfectly collinear
    price_c = 50.0 * np.cumprod(1 + np.random.normal(0.0002, 0.02, size=n_days))

    df = pd.DataFrame(
        {"ASSET_A": price_a, "ASSET_B": price_b, "ASSET_C": price_c},
        index=dates,
    )
    return MarketData(prices=df, risk_free_rate=0.02)


@pytest.fixture
def short_horizon_market_data() -> MarketData:
    """Creates MarketData where number of assets N > observations T (singular empirical covariance)."""
    np.random.seed(42)
    n_days = 5
    n_assets = 10
    assets = [f"A_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.015, size=(n_days, n_assets))
    prices = 100.0 * np.cumprod(1 + returns, axis=0)
    df = pd.DataFrame(prices, index=dates, columns=assets)
    return MarketData(prices=df, risk_free_rate=0.0)


@pytest.fixture
def static_market_data() -> MarketData:
    """Loads offline static CSV dataset for deterministic integration tests."""
    csv_path = Path(__file__).parent / "fixtures" / "sample_prices.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return MarketData(prices=df, risk_free_rate=0.02)
