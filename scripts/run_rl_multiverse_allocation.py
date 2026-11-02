"""End-to-End Institutional Reinforcement Learning & Hierarchical Meta-RL Benchmark.

Demonstrates:
1. Multiverse Scenario Generation (`GMMRegimeGenerator`) producing a 3D (M, T, N) cube.
2. Evolvable State Space (`POMDPPortfolioEnv` + `FeaturePipeline` + `DictFlattenWrapper`)
   with `AlmgrenChrissCostModel` and Moody-Saffell `DifferentialSharpeReward`.
3. Continuous-Control Policy Training (`SB3PortfolioAgent` with PPO) wrapped inside
   `RLAllocator` with thermodynamic `WassersteinRegimeGate` inertia.
4. Hierarchical Meta-RL (`MetaPortfolioEnv` + `HierarchicalMetaRLAllocator`) dynamically
   blending Markowitz, Risk Parity, HRP, and Multi-Period MPC experts.
5. Walk-Forward Evaluation & Interactive HTML Tear Sheet export (`MultiPeriodTearSheet`
   and `AllocationTearSheet`) to `visualizations_output/rl/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyalloq_backtest.costs import AlmgrenChrissCostModel, FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult
from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_features.core.pipeline import FeaturePipeline
from pyalloq_simulation import GMMRegimeGenerator
from pyalloq_viz import AllocationTearSheet, MultiPeriodTearSheet

from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.rl.agents.meta_allocator import HierarchicalMetaRLAllocator
from pyalloq.rl.agents.sb3_wrapper import SB3PortfolioAgent
from pyalloq.rl.allocator import RLAllocator
from pyalloq.rl.envs.meta_env import MetaPortfolioEnv
from pyalloq.rl.envs.pomdp_env import POMDPPortfolioEnv
from pyalloq.rl.rewards.functions import DifferentialSharpeReward
from pyalloq.rl.thermodynamics import WassersteinRegimeGate

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "EFA"]


class RollingVolMomentumTransformer(BaseFeatureTransformer):
    """Appends rolling 10-day volatility and momentum features to `MarketData`."""

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def transform(self, data: MarketData) -> MarketData:
        rets = data.prices.pct_change().fillna(0.0)
        data.features[f"vol_{self.window}"] = (
            rets.rolling(self.window, min_periods=1).std().fillna(0.0)
        )
        data.features[f"mom_{self.window}"] = (
            rets.rolling(self.window, min_periods=1).mean().fillna(0.0)
        )
        return data


def build_market_data() -> MarketData:
    """Create a realistic multi-regime 6-asset MarketData object with offline fallback."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2022-01-03", periods=360, freq="B")
    n_assets = len(TICKERS)

    # Construct two alternating market regimes (calm growth vs high-vol drawdown)
    base_mu = np.array([0.0005, 0.0006, 0.0001, 0.0003, 0.0003, 0.0004])
    crisis_mu = np.array([-0.0012, -0.0015, 0.0004, 0.0006, -0.0010, -0.0011])

    factor = rng.normal(0.0, 0.009, size=(360, 1))
    idio = rng.normal(0.0, 0.006, size=(360, n_assets))
    betas = np.array([1.0, 1.25, -0.35, 0.05, 0.85, 0.90])

    rets = np.zeros((360, n_assets), dtype=float)
    for t in range(360):
        in_stress = (110 <= t < 165) or (260 <= t < 305)
        mu_t = crisis_mu if in_stress else base_mu
        vol_mult = 1.8 if in_stress else 1.0
        rets[t] = mu_t + vol_mult * (factor[t] * betas + idio[t])

    prices_df = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=TICKERS,
    )
    return MarketData(prices=prices_df)


def main() -> None:
    print("=" * 84)
    print("PyAlloq[rl]: Multiverse POMDP & Hierarchical Meta-RL Portfolio Allocation")
    print("=" * 84)

    market_data = build_market_data()
    train_prices = market_data.prices.iloc[:180]
    train_data = MarketData(prices=train_prices)

    # 1. Generate 3D Multiverse Scenarios via GMMRegimeGenerator
    print("\n[1/4] Generating 3D Multiverse Scenarios (M=30 paths, T=90 steps)...")
    gmm_gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
    gmm_gen.fit(train_data)
    scenarios = gmm_gen.generate(n_paths=30, horizon=90)

    # 2. Build POMDPPortfolioEnv with FeaturePipeline, Almgren-Chriss & Differential Sharpe
    print(
        "[2/4] Training Continuous-Control SB3PortfolioAgent (PPO) on POMDPPortfolioEnv..."
    )
    feature_pipeline = FeaturePipeline().add(RollingVolMomentumTransformer(window=10))
    pomdp_env = POMDPPortfolioEnv(
        scenario_data=scenarios,
        feature_pipeline=feature_pipeline,
        lookback_window=20,
        cost_model=AlmgrenChrissCostModel(
            portfolio_aum=1e6, spread_bps=2.0, gamma=0.04
        ),
        reward_function=DifferentialSharpeReward(eta=0.03),
        include_cash=True,
        projection_mode="softmax",
    )

    pomdp_agent = SB3PortfolioAgent(
        env=pomdp_env,
        algorithm="PPO",
        hidden_dim=64,
        learning_rate=1e-3,
        seed=42,
    )
    pomdp_agent.fit(total_timesteps=320)

    w2_gate = WassersteinRegimeGate(
        fast_window=15, slow_window=45, w2_threshold=0.025, sensitivity=1.0
    )
    pomdp_rl_allocator = RLAllocator(
        policy=pomdp_agent,
        lookback_window=20,
        regime_gate=w2_gate,
        use_pomdp_obs=True,
        include_cash=True,
        feature_pipeline=feature_pipeline,
        tickers=TICKERS,
    )

    # 3. Train Hierarchical Meta-RL Allocator over 4 Classical/ML/MPC Experts
    print("[3/4] Training HierarchicalMetaRLAllocator over 4 Base Allocator Experts...")
    mpc_allocator = MultiPeriodMPCAllocator(
        tickers=TICKERS,
        horizon=4,
        risk_aversion=3.0,
        proportional_cost_bps=10.0,
        quadratic_cost_coeff=0.02,
        backend="admm",
    )
    base_experts = [
        MarkowitzAllocator(tickers=TICKERS, objective=ObjectiveFunction.MAX_SHARPE),
        RiskParityAllocator(tickers=TICKERS),
        HRPAllocator(tickers=TICKERS),
        mpc_allocator,
    ]
    alpha_decay_est = AlphaDecayReturnEstimator(horizon=4, alpha_decay=0.15)
    ewma_est = EWMAReturnEstimator(span=30)
    cov_est = LedoitWolfShrinkage()

    meta_env = MetaPortfolioEnv(
        scenario_data=train_data,
        base_allocators=base_experts,
        returns_estimator=ewma_est,
        cov_estimator=cov_est,
        lookback_window=30,
        rebalance_stride=10,
        cost_model=FlatBpsCostModel(bps=10.0),
        reward_function=DifferentialSharpeReward(eta=0.04),
    )
    meta_agent = SB3PortfolioAgent(
        env=meta_env,
        algorithm="PPO",
        hidden_dim=32,
        learning_rate=1e-3,
        seed=42,
    )
    meta_agent.fit(total_timesteps=120)

    meta_rl_allocator = HierarchicalMetaRLAllocator(
        base_allocators=base_experts,
        meta_agent=meta_agent,
        returns_estimator=ewma_est,
        cov_estimator=cov_est,
        lookback_window=30,
        tickers=TICKERS,
    )

    # 4. Walk-Forward Out-of-Sample Evaluation
    print("[4/4] Running Walk-Forward Evaluation across Strategies...")
    strategies = {
        "POMDP RL Agent (PPO + W2 Gate)": StrategyPipeline(
            returns_estimator=ewma_est,
            cov_estimator=cov_est,
            allocator=pomdp_rl_allocator,
        ),
        "Hierarchical Meta-RL Committee": StrategyPipeline(
            returns_estimator=ewma_est,
            cov_estimator=cov_est,
            allocator=meta_rl_allocator,
        ),
        "Operator-Split MPC (ADMM)": StrategyPipeline(
            returns_estimator=alpha_decay_est,
            cov_estimator=cov_est,
            allocator=mpc_allocator,
        ),
        "HRP Baseline": StrategyPipeline(
            returns_estimator=ewma_est,
            cov_estimator=cov_est,
            allocator=HRPAllocator(tickers=TICKERS),
        ),
    }

    cost_model = FlatBpsCostModel(bps=10.0)
    backtest_results: dict[str, BacktestResult] = {}
    summary_rows: list[dict[str, str]] = []

    for name, pipeline in strategies.items():
        engine = WalkForwardEngine(
            pipeline=pipeline,
            splitter=RollingWindowSplitter(lookback_window=60),
            rebalance_freq="ME",
            cost_model=cost_model,
        )
        res = engine.run(market_data)
        backtest_results[name] = res
        ts = res.tear_sheet["Value"]
        summary_rows.append(
            {
                "Strategy": name,
                "Ann. Return": f"{float(ts['annualized_return']):.2%}",
                "Ann. Vol": f"{float(ts['annualized_volatility']):.2%}",
                "Sharpe": f"{float(ts['sharpe_ratio']):.2f}",
                "Max Drawdown": f"{float(ts['maximum_drawdown']):.2%}",
                "Ann. Turnover": f"{float(ts.get('annualized_turnover', 0.0)):.2f}x",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nOut-of-Sample Walk-Forward Performance Summary (10 bps Flat Cost):")
    print(summary_df.to_string(index=False))

    # Export Interactive HTML Tear Sheets
    output_dir = Path("visualizations_output/rl")
    output_dir.mkdir(parents=True, exist_ok=True)

    cov_full = cov_est.estimate(market_data)
    horizon_forecasts = alpha_decay_est.estimate_horizon(market_data)
    pre_trade_w = pd.Series(1.0 / len(TICKERS), index=TICKERS)
    mpo_res = mpc_allocator.allocate(
        market_data,
        cov_matrix=cov_full,
        expected_returns=horizon_forecasts,
        current_weights=pre_trade_w,
    )

    mpo_ts = MultiPeriodTearSheet(
        mpo_result=mpo_res,
        horizon_forecasts=horizon_forecasts,
        market_data=market_data,
        strategy_comparison=backtest_results,
        current_weights=pre_trade_w,
    )
    mpo_ts_path = output_dir / "1_rl_multiverse_comparison_tearsheet.html"
    mpo_ts.to_html(path=mpo_ts_path)

    meta_opt_res = meta_rl_allocator.allocate(
        market_data,
        cov_matrix=cov_full,
        expected_returns=ewma_est.estimate(market_data),
        current_weights=pre_trade_w,
    )
    alloc_ts = AllocationTearSheet(
        optimization_result=meta_opt_res,
        backtest_result=backtest_results["Hierarchical Meta-RL Committee"],
        market_data=market_data,
        cov_matrix=cov_full,
    )
    alloc_ts_path = output_dir / "2_hierarchical_meta_rl_allocation_tearsheet.html"
    alloc_ts.to_html(path=alloc_ts_path)

    print(
        f"\nSaved Interactive RL Tear Sheets to:\n  - {mpo_ts_path}\n  - {alloc_ts_path}"
    )


if __name__ == "__main__":
    main()
