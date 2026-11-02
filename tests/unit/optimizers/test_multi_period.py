import numpy as np
import pandas as pd
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import MultiPeriodOptimizationResult

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq.classical.optimizers.stochastic_mpc import (
    StochasticScenarioMPCAllocator,
)
from pyalloq.ml.optimizers.bhrp import BayesianHRPAllocator


def _make_market_data(n_days: int = 120, n_assets: int = 4) -> MarketData:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    rets = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers
    )
    return MarketData(prices=prices)


def test_alpha_decay_return_estimator_shapes_and_decay() -> None:
    md = _make_market_data()
    est = AlphaDecayReturnEstimator(horizon=5, alpha_decay=0.2)

    s_ret = est.estimate(md)
    assert isinstance(s_ret, pd.Series)
    assert len(s_ret) == len(md.assets)

    traj_df = est.estimate_trajectory(md)
    assert isinstance(traj_df, pd.DataFrame)
    assert traj_df.shape == (5, len(md.assets))
    np.testing.assert_allclose(s_ret.values, traj_df.iloc[0].values)
    # Signal norm decays across horizon stages
    assert np.linalg.norm(traj_df.iloc[-1].values) < np.linalg.norm(
        traj_df.iloc[0].values
    )


def test_mpc_allocator_cvxpy_and_no_trade_zone() -> None:
    md = _make_market_data()
    cov = EmpiricalCovariance().estimate(md)
    mu_traj = AlphaDecayReturnEstimator(horizon=4).estimate_trajectory(md)

    current_w = pd.Series([0.4, 0.3, 0.2, 0.1], index=md.assets)

    # Low spread cost -> trades away from current_w
    mpc_low_cost = MultiPeriodMPCAllocator(
        tickers=md.assets,
        horizon=4,
        risk_aversion=2.0,
        proportional_cost_bps=1.0,
        market_impact_coeff=0.01,
        quadratic_cost_coeff=0.005,
        permanent_impact_coeff=0.01,
        backend="cvxpy",
    )
    res_low = mpc_low_cost.allocate(
        md,
        cov_matrix=cov,
        expected_returns=mu_traj,
        current_weights=current_w,
    )
    assert isinstance(res_low, MultiPeriodOptimizationResult)
    assert res_low.weights_trajectory is not None
    assert res_low.trades_trajectory is not None
    assert res_low.weights_trajectory.shape == (4, 4)
    assert res_low.trades_trajectory.shape == (4, 4)
    assert np.isclose(res_low.weights.sum(), 1.0, atol=1e-4)
    assert (res_low.weights >= -1e-5).all()

    # Very high spread cost -> stays inside the p=1 polyhedral no-trade zone around current_w
    mpc_high_spread = MultiPeriodMPCAllocator(
        tickers=md.assets,
        horizon=4,
        risk_aversion=2.0,
        proportional_cost_bps=500.0,
        backend="cvxpy",
    )
    res_high = mpc_high_spread.allocate(
        md,
        cov_matrix=cov,
        expected_returns=mu_traj,
        current_weights=current_w,
    )
    np.testing.assert_allclose(res_high.weights.values, current_w.values, atol=1e-3)


def test_mpc_allocator_admm_backend_consistency() -> None:
    md = _make_market_data()
    cov = EmpiricalCovariance().estimate(md)
    mu_traj = AlphaDecayReturnEstimator(horizon=4).estimate_trajectory(md)
    current_w = pd.Series([0.25, 0.25, 0.25, 0.25], index=md.assets)

    mpc_cvx = MultiPeriodMPCAllocator(
        tickers=md.assets,
        horizon=4,
        risk_aversion=3.0,
        proportional_cost_bps=10.0,
        quadratic_cost_coeff=0.05,
        backend="cvxpy",
    )
    mpc_admm = MultiPeriodMPCAllocator(
        tickers=md.assets,
        horizon=4,
        risk_aversion=3.0,
        proportional_cost_bps=10.0,
        quadratic_cost_coeff=0.05,
        backend="admm",
        admm_max_iter=300,
        admm_tol=1e-6,
    )

    res_cvx = mpc_cvx.allocate(
        md, cov_matrix=cov, expected_returns=mu_traj, current_weights=current_w
    )
    res_admm = mpc_admm.allocate(
        md, cov_matrix=cov, expected_returns=mu_traj, current_weights=current_w
    )

    assert np.isclose(res_admm.weights.sum(), 1.0, atol=1e-4)
    assert (res_admm.weights >= -1e-5).all()
    np.testing.assert_allclose(
        res_cvx.weights.values, res_admm.weights.values, atol=0.04
    )


def test_stochastic_scenario_mpc_allocator() -> None:
    rng = np.random.default_rng(7)
    tickers = ["A", "B", "C"]
    scen_returns = rng.normal(0.001, 0.015, size=(40, 4, 3))
    scen_prices = 100.0 * np.exp(np.cumsum(scen_returns, axis=1))
    scen_data = ScenarioMarketData(
        prices=scen_prices, returns=scen_returns, asset_names=tickers
    )

    alloc = StochasticScenarioMPCAllocator(
        tickers=tickers,
        horizon=4,
        cvar_alpha=0.10,
        cvar_weight=1.5,
        proportional_cost_bps=10.0,
    )
    current_w = pd.Series([0.5, 0.3, 0.2], index=tickers)
    res = alloc.allocate(scen_data, current_weights=current_w)

    assert isinstance(res, MultiPeriodOptimizationResult)
    assert res.weights_trajectory is not None
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
    assert res.weights_trajectory.shape == (4, 3)
    assert res.metadata["n_scenarios"] == 40


def test_bayesian_hrp_allocator_and_walkforward_stateful_weights() -> None:
    md = _make_market_data(n_days=140, n_assets=5)
    bhrp = BayesianHRPAllocator(
        tickers=md.assets,
        tau=0.3,
        return_tilt_strength=2.0,
        turnover_penalty=0.5,
    )
    pipe = StrategyPipeline(
        allocator=bhrp,
        returns_estimator=AlphaDecayReturnEstimator(horizon=3),
        cov_estimator=EmpiricalCovariance(),
    )
    engine = WalkForwardEngine(
        pipeline=pipe,
        splitter=RollingWindowSplitter(lookback_window=40),
        rebalance_freq="W-FRI",
        cost_model=FlatBpsCostModel(bps=10.0),
    )
    bt_res = engine.run(md)
    assert len(bt_res.returns) > 0
    valid_w = bt_res.weights.dropna()
    assert len(valid_w) > 0
    assert np.allclose(valid_w.sum(axis=1).values, 1.0, atol=1e-4)
