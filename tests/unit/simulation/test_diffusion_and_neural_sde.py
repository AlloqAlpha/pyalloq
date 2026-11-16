from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioEstimator
from pyalloq_simulation import ConditionalDiffusionGenerator, NeuralSDEGenerator

from pyalloq.dl.estimators import NeuralSDEEstimator


@pytest.fixture
def sample_market_data() -> MarketData:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2024-01-02", periods=90)
    assets = ["AAPL", "MSFT", "NVDA", "GOOG"]

    # Common factor + heavy-tailed shocks
    factor = rng.normal(0.0005, 0.012, size=(90, 1))
    loadings = np.array([[1.1, 0.9, 1.4, 0.85]])
    idio = rng.standard_t(df=5, size=(90, 4)) * 0.008
    rets = factor @ loadings + idio
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)

    prices_df = pd.DataFrame(prices, index=dates, columns=assets)
    iv_df = pd.DataFrame(
        0.22 + 0.05 * np.abs(rng.standard_normal(size=(90, 4))),
        index=dates,
        columns=assets,
    )
    macro_df = pd.DataFrame(
        {"vix_term": 18.0 + np.cumsum(rng.normal(0.0, 0.3, size=90))},
        index=dates,
    )
    return MarketData(
        prices=prices_df,
        features={"implied_vol": iv_df, "macro": macro_df},
        risk_free_rate=0.04,
    )


def test_conditional_diffusion_generator_end_to_end(
    sample_market_data: MarketData,
) -> None:
    gen = ConditionalDiffusionGenerator(
        n_factors=3,
        condition_window=12,
        signature_window=5,
        diffusion_steps=60,
        ddim_steps=12,
        ddim_eta=0.2,
        random_state=42,
    )
    assert isinstance(gen, BaseScenarioEstimator)

    scenarios = gen.estimate_scenarios(sample_market_data, n_paths=40, horizon=15)
    assert isinstance(scenarios, ScenarioMarketData)
    assert scenarios.prices.shape == (40, 15, 4)
    assert scenarios.returns.shape == (40, 15, 4)
    assert np.all(np.isfinite(scenarios.prices))
    assert np.all(scenarios.prices > 0.0)
    assert "implied_vol" in scenarios.features
    assert scenarios.features["implied_vol"].shape == (40, 15, 4)
    assert "macro" in scenarios.features
    assert scenarios.features["macro"].shape == (40, 15, 1)

    # Reproducibility check with identical random_state
    gen2 = ConditionalDiffusionGenerator(
        n_factors=3,
        condition_window=12,
        signature_window=5,
        diffusion_steps=60,
        ddim_steps=12,
        ddim_eta=0.2,
        random_state=42,
    )
    scen2 = gen2.estimate_scenarios(sample_market_data, n_paths=40, horizon=15)
    assert np.allclose(scenarios.returns, scen2.returns)


def test_neural_sde_generator_no_arbitrage_and_surfaces(
    sample_market_data: MarketData,
) -> None:
    max_sr = 1.25
    sde_gen = NeuralSDEGenerator(
        hidden_dim=24,
        substeps_per_day=2,
        max_market_price_of_risk=max_sr,
        signature_window=5,
        random_state=99,
    )
    sde_gen.fit(sample_market_data)
    assert sde_gen.is_fitted
    assert sde_gen.signature_mmd_loss_ >= 0.0
    assert sde_gen.last_market_price_of_risk_norm_ <= max_sr + 1e-4

    # Instantaneous t=0 covariance surface must be symmetric positive-definite
    cov_0 = sde_gen.estimate_instantaneous_covariance()
    assert cov_0.shape == (4, 4)
    assert list(cov_0.columns) == sample_market_data.assets
    eigvals = np.linalg.eigvalsh(cov_0.to_numpy(dtype=np.float64))
    assert np.all(eigvals > 0.0)

    scenarios = sde_gen.generate(n_paths=35, horizon=12)
    assert scenarios.prices.shape == (35, 12, 4)
    assert scenarios.returns.shape == (35, 12, 4)
    assert "implied_vol" in scenarios.features


def test_neural_sde_estimator_dual_role_interfaces(
    sample_market_data: MarketData,
) -> None:
    estimator = NeuralSDEEstimator(
        horizon=10,
        n_paths=48,
        hidden_dim=16,
        epochs=10,
        max_market_price_of_risk=1.5,
        random_state=7,
    )
    scenarios = estimator.estimate_scenarios(sample_market_data)
    assert isinstance(scenarios, ScenarioMarketData)
    assert scenarios.returns.shape == (48, 10, 4)
    assert np.isfinite(estimator.last_training_loss_)
    assert estimator.last_signature_mmd_ >= 0.0

    expected_rets = estimator.estimate(sample_market_data, refit=False)
    assert isinstance(expected_rets, pd.Series)
    assert list(expected_rets.index) == sample_market_data.assets
    assert np.all(np.isfinite(expected_rets.to_numpy(dtype=np.float64)))

    cov_df = estimator.estimate_covariance(sample_market_data, refit=False)
    assert isinstance(cov_df, pd.DataFrame)
    assert cov_df.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(cov_df.to_numpy(dtype=np.float64)) > 0.0)
