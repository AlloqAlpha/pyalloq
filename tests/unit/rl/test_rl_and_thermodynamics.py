from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from pyalloq_core.data import MarketData, ScenarioMarketData

from pyalloq.rl import (
    DeepHedgingTrajectoryTrainer,
    PortfolioGymEnv,
    RLAllocator,
    ScenarioPortfolioEnv,
    TrajectoryPolicyNet,
    WassersteinRegimeGate,
)

pytest.importorskip("torch")


def _make_market_data(n_days: int = 80, n_assets: int = 3) -> MarketData:
    rng = np.random.default_rng(55)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    rets = rng.normal(0.0004, 0.012, size=(n_days, n_assets))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers
    )
    return MarketData(prices=prices)


def test_wasserstein_regime_gate_distance_and_damping() -> None:
    gate = WassersteinRegimeGate(fast_window=15, slow_window=50, w2_threshold=0.01)

    # Identical distributions -> W2 = 0, Carnot efficiency = 0
    mu = np.zeros(3)
    cov = np.eye(3) * 0.0001
    w2_zero = gate.bures_wasserstein_distance(mu, cov, mu, cov)
    assert np.isclose(w2_zero, 0.0, atol=1e-5)

    # Construct regime shock in last 15 days (high vol vs calm prior)
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0005, 0.005, size=(45, 3))
    crisis = rng.normal(-0.005, 0.045, size=(15, 3))
    returns_df = pd.DataFrame(np.vstack([calm, crisis]), columns=["A", "B", "C"])

    diag = gate.evaluate(returns_df)
    assert diag.wasserstein_distance > 0.02
    assert 0.0 < diag.carnot_efficiency <= 1.0
    assert diag.damping_factor < 1.0

    target_w = pd.Series([0.8, 0.1, 0.1], index=["A", "B", "C"])
    curr_w = pd.Series([0.2, 0.4, 0.4], index=["A", "B", "C"])
    gated_w, _ = gate.gate_weights(target_w, curr_w, returns_df)
    assert np.isclose(gated_w.sum(), 1.0, atol=1e-6)
    # Gated weights are pulled closer to curr_w than raw target_w
    assert np.abs(gated_w - curr_w).sum() < np.abs(target_w - curr_w).sum()


def test_portfolio_gym_env_and_scenario_env_rollout() -> None:
    md = _make_market_data(n_days=50, n_assets=3)
    env = PortfolioGymEnv(
        data=md,
        lookback_window=10,
        transaction_cost_bps=10.0,
        turbulence_threshold=15.0,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.state_dim,)
    assert np.isclose(info["portfolio_value"], 1.0)

    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert np.isfinite(reward)
        assert np.isclose(step_info["weights"].sum(), 1.0, atol=1e-5)

    assert steps == 50 - 10

    # Test ScenarioPortfolioEnv
    scen_ret = np.random.default_rng(1).normal(0.0, 0.01, size=(10, 20, 3))
    scen_prices = 100.0 * np.exp(np.cumsum(scen_ret, axis=1))
    scen_data = ScenarioMarketData(
        prices=scen_prices, returns=scen_ret, asset_names=["T0", "T1", "T2"]
    )
    s_env = ScenarioPortfolioEnv(scen_data, lookback_window=5)
    s_obs, s_info = s_env.reset(options={"scenario_idx": 3})
    assert s_info["scenario_idx"] == 3
    s_obs2, s_rew, s_term, _, _ = s_env.step(np.array([0.5, 0.3, 0.2]))
    assert s_obs2.shape == s_obs.shape
    assert np.isfinite(s_rew)
    assert not s_term


def test_deep_hedging_trajectory_trainer_and_rl_allocator() -> None:
    torch.manual_seed(42)
    md = _make_market_data(n_days=70, n_assets=3)
    policy = TrajectoryPolicyNet(n_assets=3, lookback_window=10, hidden_dim=32)
    trainer = DeepHedgingTrajectoryTrainer(
        policy=policy,
        horizon=5,
        transaction_cost_bps=10.0,
        risk_aversion=2.0,
        lr=2e-3,
    )
    losses = trainer.fit_on_market_data(md, epochs=5, batch_size=8)
    assert len(losses) == 5
    assert all(np.isfinite(val) for val in losses)

    gate = WassersteinRegimeGate(fast_window=10, slow_window=30, w2_threshold=0.02)
    rl_alloc = RLAllocator(
        policy=policy,
        lookback_window=10,
        regime_gate=gate,
    )
    curr_w = pd.Series([0.5, 0.3, 0.2], index=md.assets)
    res = rl_alloc.allocate(md, current_weights=curr_w)
    assert res.status == "OPTIMAL_RL_POLICY"
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
    assert "wasserstein_distance" in res.metadata

    # Also verify SB3-compatible `.predict(obs, deterministic=True)` protocol
    class _DummySB3Model:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, Any]:
            return np.array([1.0, 0.0, -1.0], dtype=np.float32), None

    sb3_alloc = RLAllocator(policy=_DummySB3Model(), lookback_window=10)
    res_sb3 = sb3_alloc.allocate(md, current_weights=curr_w)
    assert np.isclose(res_sb3.weights.sum(), 1.0, atol=1e-5)
    assert res_sb3.weights.iloc[0] > res_sb3.weights.iloc[2]
