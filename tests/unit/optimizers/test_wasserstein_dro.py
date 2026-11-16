from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_simulation import ConditionalDiffusionGenerator

from pyalloq.classical.optimizers import WassersteinDROAllocator


@pytest.fixture
def dro_market_data() -> MarketData:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=100)
    assets = ["TECH", "FIN", "HLTH", "UTIL"]

    # Give TECH high mean & high volatility, UTIL lower volatility
    means = np.array([0.0014, 0.0007, 0.0006, 0.0003])
    vols = np.array([0.024, 0.015, 0.012, 0.008])
    corr = np.array(
        [
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.25],
            [0.1, 0.2, 0.25, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    rets = rng.multivariate_normal(means, cov, size=100)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return MarketData(prices=pd.DataFrame(prices, index=dates, columns=assets))


@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveFunction.WASSERSTEIN_KELLY,
        ObjectiveFunction.WASSERSTEIN_CVAR,
        ObjectiveFunction.WASSERSTEIN_MEAN_RISK,
    ],
)
@pytest.mark.parametrize("ground_metric", ["mahalanobis", "l2", "l1"])
def test_wasserstein_dro_objectives_and_metrics(
    dro_market_data: MarketData,
    objective: ObjectiveFunction,
    ground_metric: str,
) -> None:
    allocator = WassersteinDROAllocator(
        objective=objective,
        epsilon=0.005,
        ground_metric=ground_metric,  # type: ignore[arg-type]
        radius_mode="fixed",
    )
    res = allocator.allocate(dro_market_data)
    assert res.weights.shape == (4,)
    assert np.isclose(float(res.weights.sum()), 1.0, atol=1e-5)
    assert np.all(res.weights.to_numpy(dtype=np.float64) >= -1e-6)
    assert res.metadata["ground_metric"] == ground_metric
    assert res.metadata["objective"] == objective.name


def test_wasserstein_kelly_robustness_against_ambiguity_radius(
    dro_market_data: MarketData,
) -> None:
    alloc_low = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_KELLY,
        epsilon=1e-5,
        ground_metric="l2",
        radius_mode="fixed",
    )
    alloc_high = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_KELLY,
        epsilon=0.05,
        ground_metric="l2",
        radius_mode="fixed",
    )
    w_low = alloc_low.allocate(dro_market_data).weights.to_numpy(dtype=np.float64)
    w_high = alloc_high.allocate(dro_market_data).weights.to_numpy(dtype=np.float64)

    # Larger Wasserstein L2 radius penalizes ||w||_2 and reduces weight concentration (HHI)
    hhi_low = float(np.sum(w_low**2))
    hhi_high = float(np.sum(w_high**2))
    assert hhi_high <= hhi_low + 1e-4


def test_fonseca_junca_expected_value_view_constraints(
    dro_market_data: MarketData,
) -> None:
    # Base allocation without expected-value views
    base_alloc = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_MEAN_RISK,
        epsilon=0.005,
        ground_metric="mahalanobis",
        radius_mode="fixed",
        risk_aversion=0.5,
    )
    res_base = base_alloc.allocate(dro_market_data)

    # Express a strong bullish Fonseca & Junca (2021) expectation view on UTIL (index 3)
    view_G = np.array([[0.0, 0.0, 0.0, 1.0]])
    view_q = np.array([0.0040])  # +40 bps/day expected return on UTIL
    view_alloc = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_MEAN_RISK,
        epsilon=0.005,
        ground_metric="mahalanobis",
        radius_mode="fixed",
        risk_aversion=0.5,
        view_matrix=view_G,
        view_targets=view_q,
        view_uncertainty=1e-2,
    )
    res_view = view_alloc.allocate(dro_market_data)

    assert float(res_view.weights["UTIL"]) > float(res_base.weights["UTIL"])
    assert res_view.metadata["active_view_multipliers"] is not None


def test_dynamic_regime_adaptive_radius_and_diffusion_scenarios(
    dro_market_data: MarketData,
) -> None:
    alloc = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_CVAR,
        epsilon=0.002,
        radius_mode="regime_adaptive",
        regime_sensitivity=0.5,
    )
    res_calm = alloc.allocate(dro_market_data)

    # Inject a structural volatility/drift break in the recent 20 days
    shock_prices = dro_market_data.prices.copy()
    rets = shock_prices.pct_change().fillna(0.0)
    rets.iloc[-20:] = rets.iloc[-20:] * 3.5 - 0.015
    shock_md = MarketData(
        prices=100.0 * (1.0 + rets).cumprod(),
    )
    res_shock = alloc.allocate(shock_md)

    assert (
        res_shock.metadata["wasserstein_regime_distance"]
        > res_calm.metadata["wasserstein_regime_distance"]
    )
    assert (
        res_shock.metadata["effective_epsilon"] > res_calm.metadata["effective_epsilon"]
    )

    # Test integration with 3D ScenarioMarketData from ConditionalDiffusionGenerator
    diff_gen = ConditionalDiffusionGenerator(
        n_factors=2, diffusion_steps=40, ddim_steps=10, random_state=11
    )
    scenarios = diff_gen.estimate_scenarios(dro_market_data, n_paths=50, horizon=10)
    res_scen = alloc.allocate(scenarios)
    assert np.isclose(float(res_scen.weights.sum()), 1.0, atol=1e-5)
    assert res_scen.metadata["n_atoms"] == 50

    # Native Generative-WDRO pipeline (arXiv:2604.04342)
    g_wdro = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_KELLY,
        epsilon=0.002,
        scenario_generator=diff_gen,
        generator_n_paths=40,
        generator_horizon=8,
        generator_mmd_sensitivity=0.25,
    )
    res_gwdro = g_wdro.allocate(dro_market_data)
    assert res_gwdro.metadata["generative_wdro"] is True
    assert res_gwdro.metadata["generator_signature_mmd"] > 0.0
    assert res_gwdro.metadata["n_atoms"] == 40


def test_exploratory_regime_mv_allocator_and_wonham_filter(
    dro_market_data: MarketData,
) -> None:
    from pyalloq.rl import ExploratoryRegimeMVAllocator, WonhamRegimeFilter

    returns_df = dro_market_data.prices.pct_change().dropna()
    w_filter = WonhamRegimeFilter(n_regimes=2, random_state=42)
    beliefs = w_filter.filter_series(returns_df)
    assert beliefs.shape == (len(returns_df), 2)
    assert np.allclose(beliefs.sum(axis=1).to_numpy(), 1.0, atol=1e-5)

    emv = ExploratoryRegimeMVAllocator(
        n_regimes=2,
        risk_aversion=2.0,
        entropy_temperature=0.05,
        turnover_penalty=0.02,
        random_state=42,
    )
    res_det = emv.allocate(dro_market_data)
    assert np.isclose(float(res_det.weights.sum()), 1.0, atol=1e-5)
    assert emv.last_policy_cov_ is not None
    assert emv.last_policy_cov_.shape == (4, 4)
    assert np.isfinite(emv.last_policy_entropy_)
    assert res_det.metadata["exploratory_sampled"] is False

    w_sampled = emv.sample_action(dro_market_data, current_weights=res_det.weights)
    assert np.isclose(float(w_sampled.sum()), 1.0, atol=1e-5)
    assert np.all(w_sampled.to_numpy(dtype=np.float64) >= 0.0)
