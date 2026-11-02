"""
End-to-End Multi-Period Portfolio Optimization (MPO) Benchmark & Demonstration Script.

Compares all 3 Pillars of PyAlloq's Multi-Period Optimization Architecture under
realistic transaction costs (15 bps proportional spread) inside `WalkForwardEngine`:
1. Single-Period Baseline (`MarkowitzAllocator`)
2. Pillar 1: Classical Convex MPC (`MultiPeriodMPCAllocator` with `AlphaDecayReturnEstimator` — both CVXPY & ADMM)
3. Pillar 2: Decision-Focused Learning (`IPMOAllocator` with `MDFPOptimizationLayer` & `HybridDFLLoss`)
4. Pillar 3: Two-Level Bayesian HRP (`BayesianHRPAllocator`) & Deep Hedging RL (`RLAllocator` + `WassersteinRegimeGate`)
"""

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline

from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq.dl.estimators.e2e.ipmo_allocator import (
    IPMOAllocator,
    MultiHorizonPredictorNet,
)
from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer
from pyalloq.dl.loss_functions.dfl import HybridDFLLoss
from pyalloq.ml.optimizers.bhrp import BayesianHRPAllocator
from pyalloq.rl import (
    DeepHedgingTrajectoryTrainer,
    RLAllocator,
    TrajectoryPolicyNet,
    WassersteinRegimeGate,
)

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ"]


def build_synthetic_market_data(
    tickers: list[str], n_days: int = 320, seed: int = 42
) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    n_assets = len(tickers)

    # Generate correlated multi-regime asset returns with mean-reverting + momentum signals
    base_corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(base_corr, 1.0)
    vols = np.linspace(0.008, 0.016, n_assets)
    cov = np.outer(vols, vols) * base_corr
    chol = np.linalg.cholesky(cov)

    drifts = np.linspace(0.0002, 0.0006, n_assets)
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    returns = drifts + shocks

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

    ret_df = prices_df.pct_change().fillna(0.0)
    mom_10 = ret_df.rolling(10, min_periods=1).mean().fillna(0.0)
    vol_10 = ret_df.rolling(10, min_periods=1).std().fillna(0.0)

    return MarketData(
        prices=prices_df,
        features={"mom_10": mom_10, "vol_10": vol_10},
    )


def train_ipmo_predictor(
    market_data: MarketData,
    horizon: int = 5,
    lookback: int = 20,
    epochs: int = 8,
) -> tuple[MultiHorizonPredictorNet, MDFPOptimizationLayer]:
    torch.manual_seed(42)
    n_assets = len(market_data.assets)
    n_features = 1 + (len(market_data.features) if market_data.features else 0)

    predictor = MultiHorizonPredictorNet(
        n_assets=n_assets,
        n_features=n_features,
        horizon=horizon,
        d_model=16,
        n_heads=2,
    )
    mdfp_layer = MDFPOptimizationLayer(
        horizon=horizon,
        risk_aversion=2.5,
        cost_penalty=0.0015,
        max_iter=30,
        implicit_backward=True,
    )
    loss_fn = HybridDFLLoss(alpha=0.7, cost_penalty=0.0015, risk_aversion=2.5)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=2e-3)

    ret_df = market_data.prices.pct_change().fillna(0.0)
    ret_np = ret_df.to_numpy(dtype=np.float32)

    feat_arrays = [ret_np[..., None]]
    if market_data.features:
        for _, fdf in market_data.features.items():
            feat_arrays.append(fdf.to_numpy(dtype=np.float32)[..., None])
    all_x = np.concatenate(feat_arrays, axis=-1)  # (T, N, F)

    # Train on the first 160 observations
    train_limit = min(160, len(ret_np) - horizon - 1)
    x_list = []
    y_list = []
    for t in range(lookback, train_limit):
        x_list.append(all_x[t - lookback : t])
        y_list.append(ret_np[t : t + horizon])

    x_batch = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)
    y_batch = torch.tensor(np.stack(y_list, axis=0), dtype=torch.float32)

    predictor.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mu_hat = predictor(x_batch)
        z_traj = mdfp_layer(mu_hat)
        loss = loss_fn(z_traj, mu_hat, y_batch)
        loss.backward()
        optimizer.step()

    predictor.eval()
    return predictor, mdfp_layer


def main() -> None:
    print("=" * 80)
    print("PyAlloq Multi-Period Portfolio Optimization (3-Pillar Benchmark)")
    print("=" * 80)

    market_data = build_synthetic_market_data(TICKERS, n_days=300)
    cost_model = FlatBpsCostModel(bps=15.0)
    cov_est = LedoitWolfShrinkage()
    alpha_decay_est = AlphaDecayReturnEstimator(horizon=5, alpha_decay=0.15)

    # 1. Single-Period Markowitz Baseline
    spo_mvo = StrategyPipeline(
        allocator=MarkowitzAllocator(
            tickers=TICKERS, objective=ObjectiveFunction.MAX_RETURN
        ),
        returns_estimator=EWMAReturnEstimator(span=40),
        cov_estimator=cov_est,
    )

    # 2. Pillar 1a: Classical Convex MPC (CVXPY with p=1 spread, p=1.5 impact, Gârleanu-Pedersen permanent impact)
    mpc_cvxpy = StrategyPipeline(
        allocator=MultiPeriodMPCAllocator(
            tickers=TICKERS,
            horizon=5,
            risk_aversion=3.0,
            proportional_cost_bps=15.0,
            market_impact_coeff=0.02,
            quadratic_cost_coeff=0.01,
            permanent_impact_coeff=0.01,
            backend="cvxpy",
        ),
        returns_estimator=alpha_decay_est,
        cov_estimator=cov_est,
    )

    # 3. Pillar 1b: High-Speed Operator Splitting MPC (ADMM)
    mpc_admm = StrategyPipeline(
        allocator=MultiPeriodMPCAllocator(
            tickers=TICKERS,
            horizon=5,
            risk_aversion=3.0,
            proportional_cost_bps=15.0,
            quadratic_cost_coeff=0.02,
            backend="admm",
        ),
        returns_estimator=alpha_decay_est,
        cov_estimator=cov_est,
    )

    # 4. Pillar 2: Decision-Focused Learning IPMO (MDFP Implicit Layer)
    predictor, mdfp_layer = train_ipmo_predictor(market_data, horizon=5, lookback=20)
    ipmo_pipeline = StrategyPipeline(
        allocator=IPMOAllocator(
            predictor=predictor,
            mdfp_layer=mdfp_layer,
            lookback_window=20,
            tickers=TICKERS,
        ),
        cov_estimator=cov_est,
    )

    # 5. Pillar 3a: Two-Level Bayesian HRP (RL-BHRP)
    bhrp_pipeline = StrategyPipeline(
        allocator=BayesianHRPAllocator(
            tickers=TICKERS,
            tau=0.3,
            return_tilt_strength=2.0,
            turnover_penalty=0.6,
        ),
        returns_estimator=alpha_decay_est,
        cov_estimator=cov_est,
    )

    # 6. Pillar 3b: Deep Hedging Trajectory Policy + Bures-Wasserstein Regime Gate
    policy = TrajectoryPolicyNet(n_assets=len(TICKERS), lookback_window=20)
    dh_trainer = DeepHedgingTrajectoryTrainer(
        policy=policy,
        horizon=5,
        transaction_cost_bps=15.0,
        risk_aversion=2.5,
    )
    train_slice = MarketData(prices=market_data.prices.iloc[:160])
    dh_trainer.fit_on_market_data(train_slice, epochs=10, batch_size=16)

    rl_pipeline = StrategyPipeline(
        allocator=RLAllocator(
            policy=policy,
            lookback_window=20,
            regime_gate=WassersteinRegimeGate(
                fast_window=15, slow_window=45, w2_threshold=0.025
            ),
            tickers=TICKERS,
        ),
    )

    strategies = {
        "Single-Period Markowitz": spo_mvo,
        "Pillar 1: Convex MPC (CVXPY)": mpc_cvxpy,
        "Pillar 1: Operator-Split MPC (ADMM)": mpc_admm,
        "Pillar 2: DFL IPMO (MDFP Layer)": ipmo_pipeline,
        "Pillar 3: Bayesian HRP (RL-BHRP)": bhrp_pipeline,
        "Pillar 3: Deep Hedging + W2 Gate": rl_pipeline,
    }

    summary_rows = []
    for name, pipe in strategies.items():
        engine = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=60),
            rebalance_freq="W-FRI",
            cost_model=cost_model,
        )
        res = engine.run(market_data)
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
    print(
        "\nWalk-Forward Net-of-Cost Performance Comparison (15 bps Transaction Cost):"
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
