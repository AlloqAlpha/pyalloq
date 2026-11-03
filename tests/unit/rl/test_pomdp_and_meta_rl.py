import numpy as np
import pandas as pd
import pytest
from pyalloq_backtest.costs import AlmgrenChrissCostModel, FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_features.core.pipeline import FeaturePipeline

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.rl.agents.meta_allocator import HierarchicalMetaRLAllocator
from pyalloq.rl.agents.sb3_wrapper import SB3PortfolioAgent
from pyalloq.rl.allocator import RLAllocator
from pyalloq.rl.envs.meta_env import MetaPortfolioEnv
from pyalloq.rl.envs.pomdp_env import POMDPPortfolioEnv
from pyalloq.rl.rewards.functions import (
    CARAUtilReward,
    DifferentialSharpeReward,
    StepwiseSortinoReward,
)
from pyalloq.rl.wrappers.dict_features import (
    DictFlattenWrapper,
    FeatureNormalizationWrapper,
)

pytest.importorskip("torch")


class _RollingVolTransformer(BaseFeatureTransformer):
    """Simple rolling volatility feature transformer for testing `FeaturePipeline`."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def transform(self, data: MarketData) -> MarketData:
        rets = data.prices.pct_change().fillna(0.0)
        data.features[f"vol_{self.window}"] = (
            rets.rolling(self.window, min_periods=1).std().fillna(0.0)
        )
        return data


@pytest.fixture
def synthetic_scenarios() -> ScenarioMarketData:
    rng = np.random.default_rng(42)
    m_paths, t_steps, n_assets = 4, 40, 3
    rets = rng.normal(0.0005, 0.01, size=(m_paths, t_steps, n_assets))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=1)
    return ScenarioMarketData(
        prices=prices,
        returns=rets,
        asset_names=["SPY", "TLT", "GLD"],
    )


@pytest.fixture
def synthetic_market_data() -> MarketData:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rets = rng.normal(0.0004, 0.01, size=(100, 3))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=["SPY", "TLT", "GLD"],
    )
    return MarketData(prices=prices)


def test_reward_functions_lifecycle() -> None:
    dsr = DifferentialSharpeReward(eta=0.05)
    sortino = StepwiseSortinoReward(window=10)
    cara = CARAUtilReward(risk_aversion=3.0)

    dsr.reset()
    sortino.reset()
    cara.reset()

    rng = np.random.default_rng(7)
    for _ in range(15):
        net_ret = float(rng.normal(0.001, 0.01))
        r_dsr = dsr.compute(
            net_return=net_ret,
            gross_return=net_ret + 0.0002,
            turnover=0.1,
            transaction_cost=0.0002,
            unrealized_drawdown=-0.02,
            w2_distance=0.01,
        )
        r_sort = sortino.compute(
            net_return=net_ret,
            gross_return=net_ret + 0.0002,
            turnover=0.1,
            transaction_cost=0.0002,
            unrealized_drawdown=-0.02,
        )
        r_cara = cara.compute(
            net_return=net_ret,
            gross_return=net_ret + 0.0002,
            turnover=0.1,
            transaction_cost=0.0002,
            unrealized_drawdown=-0.02,
            w2_distance=0.01,
        )
        assert np.isfinite(r_dsr)
        assert np.isfinite(r_sort)
        assert np.isfinite(r_cara)


def test_pomdp_env_dict_space_and_feature_pipeline(
    synthetic_scenarios: ScenarioMarketData,
) -> None:
    fp = FeaturePipeline().add(_RollingVolTransformer(window=5))
    cost_model = AlmgrenChrissCostModel(portfolio_aum=1e6, spread_bps=2.0, gamma=0.05)
    reward_fn = DifferentialSharpeReward(eta=0.05)

    env = POMDPPortfolioEnv(
        scenario_data=synthetic_scenarios,
        feature_pipeline=fp,
        lookback_window=10,
        cost_model=cost_model,
        reward_function=reward_fn,
        include_cash=True,
        projection_mode="softmax",
    )

    obs, info = env.reset(seed=42)
    assert set(obs.keys()) == {"market_state", "current_weights", "portfolio_meta"}
    # 1 base return channel + 1 _RollingVolTransformer feature = 2 features
    assert obs["market_state"].shape == (10, 3, 2)
    # current_weights tracks the N risky asset weights
    assert obs["current_weights"].shape == (3,)
    assert obs["portfolio_meta"].shape == (3,)
    assert "scenario_idx" in info

    # Verify lazy cache populated for the sampled path
    assert info["scenario_idx"] in env._path_tensor_cache

    # Test step and simplex projection across all 3 projection modes
    for proj in ("softmax", "dirichlet", "relu_l1"):
        env.projection_mode = proj  # type: ignore[assignment]
        raw_action = np.array([0.5, -0.2, 1.2, 0.1], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(raw_action)
        assert np.isfinite(reward)
        assert not truncated
        total_w = float(np.sum(obs["current_weights"])) + float(
            step_info["cash_reserve"]
        )
        assert np.isclose(total_w, 1.0, atol=1e-4)
        assert np.all(obs["current_weights"] >= 0.0)
        assert step_info["transaction_cost"] >= 0.0


def test_dict_flatten_and_normalization_wrappers(
    synthetic_scenarios: ScenarioMarketData,
) -> None:
    env = POMDPPortfolioEnv(
        scenario_data=synthetic_scenarios,
        lookback_window=8,
        include_cash=False,
    )
    norm_env = FeatureNormalizationWrapper(env)
    flat_env = DictFlattenWrapper(norm_env)

    expected_dim = 8 * 3 * 1 + 3 + 3
    assert flat_env.observation_space.shape == (expected_dim,)

    flat_obs, _ = flat_env.reset(seed=42)
    assert flat_obs.shape == (expected_dim,)
    assert flat_obs.dtype == np.float32

    unflattened = flat_env.unflatten(flat_obs)
    assert unflattened["market_state"].shape == (8, 3, 1)
    assert unflattened["current_weights"].shape == (3,)
    assert unflattened["portfolio_meta"].shape == (3,)


def test_sb3_portfolio_agent_and_rl_allocator(
    synthetic_scenarios: ScenarioMarketData,
    synthetic_market_data: MarketData,
    tmp_path: pytest.TempPathFactory,
) -> None:
    env = POMDPPortfolioEnv(
        scenario_data=synthetic_scenarios,
        lookback_window=10,
        include_cash=False,
        cost_model=FlatBpsCostModel(bps=5.0),
        reward_function=StepwiseSortinoReward(),
    )
    agent = SB3PortfolioAgent(
        env=env,
        algorithm="PPO",
        hidden_dim=32,
        learning_rate=1e-3,
        seed=42,
    )
    agent.fit(total_timesteps=40)

    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))

    # Test save & load
    save_file = str(tmp_path / "sb3_agent_test.pt")  # type: ignore[operator]
    agent.save(save_file)
    agent.load(save_file)

    # Integrate into RLAllocator with POMDP dict observation
    rl_alloc = RLAllocator(
        policy=agent,
        lookback_window=10,
        use_pomdp_obs=True,
        include_cash=False,
    )
    res = rl_alloc.allocate(synthetic_market_data)
    assert len(res.weights) == 3
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
    assert (res.weights >= 0.0).all()


def test_meta_portfolio_env_and_hierarchical_meta_rl_allocator(
    synthetic_market_data: MarketData,
) -> None:
    tickers = ["SPY", "TLT", "GLD"]
    base_allocators = [
        RiskParityAllocator(tickers=tickers),
        HRPAllocator(tickers=tickers),
        MarkowitzAllocator(tickers=tickers, objective=ObjectiveFunction.MAX_SHARPE),
    ]
    ret_est = EWMAReturnEstimator(span=20)
    cov_est = EmpiricalCovariance()

    meta_env = MetaPortfolioEnv(
        scenario_data=synthetic_market_data,
        base_allocators=base_allocators,
        returns_estimator=ret_est,
        cov_estimator=cov_est,
        lookback_window=20,
        rebalance_stride=5,
    )
    obs, info = meta_env.reset(seed=42)
    # 3 base allocators + [vol, mom] = 5 features across 3 assets
    assert obs["market_state"].shape == (1, 3, 5)
    assert obs["current_weights"].shape == (3,)
    assert obs["portfolio_meta"].shape == (3,)
    assert "scenario_idx" in info

    meta_agent = SB3PortfolioAgent(
        env=meta_env,
        algorithm="PPO",
        hidden_dim=16,
        seed=42,
    )
    meta_agent.fit(total_timesteps=25)

    meta_allocator = HierarchicalMetaRLAllocator(
        base_allocators=base_allocators,
        meta_agent=meta_agent,
        returns_estimator=ret_est,
        cov_estimator=cov_est,
        lookback_window=20,
        tickers=tickers,
    )

    pipeline = StrategyPipeline(
        returns_estimator=ret_est,
        cov_estimator=cov_est,
        allocator=meta_allocator,
    )
    engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=RollingWindowSplitter(lookback_window=30),
        rebalance_freq="ME",
    )
    bt_res = engine.run(synthetic_market_data)
    active_weights = bt_res["weights"].dropna()
    assert not active_weights.empty
    assert np.allclose(active_weights.sum(axis=1).values, 1.0, atol=1e-4)

    # Also test standalone allocate() metadata
    opt_res = meta_allocator.allocate(synthetic_market_data)
    assert "allocator_blend_weights" in opt_res.metadata
