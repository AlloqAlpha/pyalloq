from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pyalloq_backtest import BenchmarkSuite
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.overfitting import CPCVSplitter, PBOEvaluator
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult
from pyalloq_simulation import (
    AbsoluteView,
    EntropyPooler,
    GMMRegimeGenerator,
    ViewBuilder,
)
from pyalloq_viz import (
    AllocationTearSheet,
    ComparisonTearSheet,
    MacroRegimeTearSheet,
    MultiPeriodTearSheet,
    RobustnessTearSheet,
    SimulationTearSheet,
    plot_alpha_decay_surface,
    plot_attribution_waterfall,
    plot_correlation_and_periodic_returns,
    plot_cpcv_paths,
    plot_cvar_tail_risk,
    plot_drawdown_fan_chart,
    plot_entropy_pooling,
    plot_equity_drawdown_overlay,
    plot_fan_chart,
    plot_hierarchical_composition,
    plot_mpc_trajectory,
    plot_mpo_pillar_comparison,
    plot_multi_strategy_fan_chart,
    plot_multi_turnover_drift,
    plot_multi_universe_frontier,
    plot_pbo_distribution,
    plot_regime_transitions,
    plot_risk_contributions,
    plot_terminal_wealth_ruin,
    plot_thermodynamic_regime_gate,
    plot_weight_drift_turnover,
)

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq.classical.optimizers.risk_budgeting import RiskBudgetingAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.ml.optimizers.nco import NCOAllocator


@pytest.fixture
def viz_market_data() -> MarketData:
    np.random.seed(42)
    n_days = 180
    assets = ["AAPL", "MSFT", "TLT", "GLD"]
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rets = np.random.normal(0.0006, 0.014, size=(n_days, len(assets)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=assets
    )
    cross_sec = pd.DataFrame(
        {
            "sector": [
                "Equities - Tech",
                "Equities - Tech",
                "Fixed Income",
                "Commodities",
            ]
        },
        index=assets,
    )
    return MarketData(prices=prices, cross_sectional=cross_sec, risk_free_rate=0.02)


@pytest.fixture
def viz_scenario_data(viz_market_data: MarketData) -> ScenarioMarketData:
    gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
    gen.fit(viz_market_data)
    return gen.generate(n_paths=60, horizon=30)


@pytest.fixture
def viz_backtest_result(viz_market_data: MarketData) -> BacktestResult:
    cov_est = EmpiricalCovariance()
    hrp = HRPAllocator(tickers=viz_market_data.assets)
    pipe = StrategyPipeline(allocator=hrp, cov_estimator=cov_est)
    engine = WalkForwardEngine(
        pipeline=pipe,
        splitter=RollingWindowSplitter(lookback_window=40),
        rebalance_freq="ME",
    )
    return engine.run(viz_market_data)


def test_simulation_plots_and_tearsheet(
    viz_scenario_data: ScenarioMarketData,
    viz_backtest_result: BacktestResult,
    tmp_path: Path,
) -> None:
    builder = ViewBuilder(viz_scenario_data)
    A, b, C, d = builder.build(
        [AbsoluteView(asset="AAPL", operator=">=", target=0.005)]
    )
    ep_res = EntropyPooler(n_scenarios=viz_scenario_data.n_paths).fit(
        A_ineq=A, b_ineq=b, C_eq=C, d_eq=d
    )

    fig_fan = plot_fan_chart(
        viz_scenario_data,
        entropy_result=ep_res,
        benchmark=viz_backtest_result,
        theme="alloq_dark",
    )
    assert isinstance(fig_fan, go.Figure)
    assert len(fig_fan.data) > 5

    fig_ruin = plot_terminal_wealth_ruin(
        viz_scenario_data,
        entropy_result=ep_res,
        ruin_threshold=-0.10,
        theme="alloq_light",
    )
    assert isinstance(fig_ruin, go.Figure)
    assert len(fig_ruin.data) >= 2

    fig_cvar = plot_cvar_tail_risk(viz_scenario_data, entropy_result=ep_res, alpha=0.05)
    assert isinstance(fig_cvar, go.Figure)
    assert len(fig_cvar.data) >= 3

    fig_dd = plot_drawdown_fan_chart(viz_scenario_data, entropy_result=ep_res)
    assert isinstance(fig_dd, go.Figure)
    assert len(fig_dd.data) >= 4

    ts = SimulationTearSheet(
        viz_scenario_data,
        entropy_result=ep_res,
        ruin_threshold=-0.10,
        benchmark=viz_backtest_result,
    )
    kpis = ts.get_kpis()
    assert "Synthetic Paths (M)" in kpis
    assert "Expected Horizon Return" in kpis

    combined_fig = ts.to_figure(theme="alloq_dark")
    assert isinstance(combined_fig, go.Figure)

    html_file = tmp_path / "simulation_tearsheet.html"
    html_str = ts.to_html(path=html_file, theme="alloq_dark")
    assert html_file.exists()
    assert "Multiverse & Simulation Tear Sheet" in html_str


def test_robustness_pbo_and_cpcv_tearsheet(
    viz_market_data: MarketData,
    viz_backtest_result: BacktestResult,
    tmp_path: Path,
) -> None:
    # 1. Test CSCV multi-strategy matrix mode
    np.random.seed(99)
    trial_returns = pd.DataFrame(
        np.random.normal(0.0004, 0.012, size=(160, 10)),
        columns=[f"Strategy_{i}" for i in range(10)],
    )
    pbo_eval = PBOEvaluator(n_partitions=6)
    pbo_cscv = pbo_eval.evaluate_cscv(trial_returns)
    assert 0.0 <= pbo_cscv.pbo <= 1.0
    assert pbo_cscv.mode == "cscv"

    fig_pbo_cscv = plot_pbo_distribution(pbo_cscv)
    assert isinstance(fig_pbo_cscv, go.Figure)

    # 2. Test CPCV Splitter on MarketData and timeline scrambling
    splitter = CPCVSplitter(n_groups=5, k_test_groups=2, max_paths=30)
    splits = splitter.split(viz_market_data)
    assert len(splits) == 10  # C(5, 2) = 10

    cpcv_res = splitter.scramble_timelines(viz_backtest_result)
    assert cpcv_res.equity_curves.shape[1] == 30
    assert len(cpcv_res.sharpe_ratios) == 30

    fig_cpcv = plot_cpcv_paths(cpcv_res)
    assert isinstance(fig_cpcv, go.Figure)

    # 3. Test Single-Strategy Benchmark PBO mode & RobustnessTearSheet
    ts_rob = RobustnessTearSheet(viz_backtest_result, cpcv_or_synthetic=cpcv_res)
    assert "PBO (Overfitting Prob)" in ts_rob.get_kpis()
    assert isinstance(ts_rob.to_figure(), go.Figure)

    html_path = tmp_path / "robustness_tearsheet.html"
    ts_rob.to_html(path=html_path, theme="alloq_light")
    assert html_path.exists()


def test_macro_regime_and_entropy_tearsheet(
    viz_market_data: MarketData,
    viz_backtest_result: BacktestResult,
    tmp_path: Path,
) -> None:
    gmm_gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
    gmm_gen.fit(viz_market_data)
    assert gmm_gen.historical_regime_probs is not None
    assert gmm_gen.historical_regime_probs.shape[1] == 2

    scen = gmm_gen.generate(n_paths=50, horizon=25)
    assert "regime_states" in scen.features

    builder = ViewBuilder(scen)
    A, b, C, d = builder.build([AbsoluteView(asset="MSFT", operator=">=", target=0.01)])
    ep_res = EntropyPooler(n_scenarios=scen.n_paths).fit(
        A_ineq=A, b_ineq=b, C_eq=C, d_eq=d
    )

    fig_ep = plot_entropy_pooling(ep_res, scenario_data=scen)
    assert isinstance(fig_ep, go.Figure)

    # Historical regime overlay
    fig_reg_hist = plot_regime_transitions(
        viz_backtest_result, gmm_or_probs=gmm_gen, market_data=viz_market_data
    )
    assert isinstance(fig_reg_hist, go.Figure)

    # Synthetic multiverse regime overlay
    fig_reg_sim = plot_regime_transitions(scen, gmm_or_probs=gmm_gen)
    assert isinstance(fig_reg_sim, go.Figure)

    ts_macro = MacroRegimeTearSheet(
        entropy_result=ep_res,
        equity_or_scenarios=viz_backtest_result,
        gmm_or_probs=gmm_gen,
        scenario_data=scen,
        market_data=viz_market_data,
    )
    assert "KL Divergence" in ts_macro.get_kpis()
    assert isinstance(ts_macro.to_figure(), go.Figure)

    html_path = tmp_path / "macro_regime_tearsheet.html"
    ts_macro.to_html(path=html_path)
    assert html_path.exists()


def test_allocation_mechanics_and_risk_tearsheet(
    viz_market_data: MarketData,
    viz_scenario_data: ScenarioMarketData,
    viz_backtest_result: BacktestResult,
    tmp_path: Path,
) -> None:
    rets = viz_market_data.get_returns()
    cov = rets.cov() * 252.0
    mu = rets.mean() * 252.0

    hrp = HRPAllocator(tickers=viz_market_data.assets)
    hrp_res = hrp.allocate(viz_market_data, cov_matrix=cov)
    assert "clusters" in hrp_res.metadata
    assert "linkage_matrix" in hrp_res.metadata

    nco = NCOAllocator(tickers=viz_market_data.assets, n_clusters=2)
    nco_res = nco.allocate(viz_market_data, cov_matrix=cov)
    assert "clusters" in nco_res.metadata

    rb = RiskBudgetingAllocator(tickers=viz_market_data.assets)
    rb_res = rb.allocate(viz_market_data, cov_matrix=cov)

    fig_ef = plot_multi_universe_frontier(
        expected_returns=mu,
        cov_matrix=cov,
        scenario_data=viz_scenario_data,
        allocator_or_pipeline=hrp,
        optimal_portfolios=[hrp_res, rb_res],
        n_random_portfolios=100,
        max_scenario_solves=15,
    )
    assert isinstance(fig_ef, go.Figure)

    fig_wd = plot_weight_drift_turnover(
        viz_backtest_result, market_data=viz_market_data
    )
    assert isinstance(fig_wd, go.Figure)

    fig_hc = plot_hierarchical_composition(
        hrp_res, target_or_benchmark_weights=rb_res, market_data=viz_market_data
    )
    assert isinstance(fig_hc, go.Figure)

    fig_rc = plot_risk_contributions(rb_res, cov_matrix=cov)
    assert isinstance(fig_rc, go.Figure)

    fig_corr = plot_correlation_and_periodic_returns(
        viz_market_data, portfolio_returns=viz_backtest_result
    )
    assert isinstance(fig_corr, go.Figure)

    ts_alloc = AllocationTearSheet(
        optimization_result=hrp_res,
        backtest_result=viz_backtest_result,
        market_data=viz_market_data,
        cov_matrix=cov,
        expected_returns=mu,
        scenario_data=viz_scenario_data,
        allocator_or_pipeline=hrp,
        target_weights=rb_res,
    )
    kpis = ts_alloc.get_kpis()
    assert "Effective Bets (1/∑w²)" in kpis
    assert "Effective Risk Bets (ENCB)" in kpis
    assert isinstance(ts_alloc.to_figure(), go.Figure)

    html_path = tmp_path / "allocation_tearsheet.html"
    ts_alloc.to_html(path=html_path)
    assert html_path.exists()


def test_multi_period_optimization_tearsheet(
    viz_market_data: MarketData,
    viz_backtest_result: BacktestResult,
    tmp_path: Path,
) -> None:
    cov = EmpiricalCovariance().estimate(viz_market_data)
    alpha_est = AlphaDecayReturnEstimator(horizon=5, alpha_decay=0.15)
    mu_traj = alpha_est.estimate_horizon(viz_market_data)

    mpc = MultiPeriodMPCAllocator(
        tickers=viz_market_data.assets,
        horizon=5,
        risk_aversion=3.0,
        proportional_cost_bps=15.0,
        market_impact_coeff=0.02,
        backend="cvxpy",
    )
    curr_w = pd.Series(1.0 / len(viz_market_data.assets), index=viz_market_data.assets)
    mpo_res = mpc.allocate(
        viz_market_data,
        cov_matrix=cov,
        expected_returns=mu_traj,
        current_weights=curr_w,
    )

    fig_traj = plot_mpc_trajectory(mpo_res, current_weights=curr_w)
    assert isinstance(fig_traj, go.Figure)

    fig_alpha = plot_alpha_decay_surface(mu_traj)
    assert isinstance(fig_alpha, go.Figure)

    fig_thermo = plot_thermodynamic_regime_gate(viz_market_data)
    assert isinstance(fig_thermo, go.Figure)

    fig_comp = plot_mpo_pillar_comparison({"MPC": viz_backtest_result})
    assert isinstance(fig_comp, go.Figure)

    ts_mpo = MultiPeriodTearSheet(
        mpo_result=mpo_res,
        horizon_forecasts=mu_traj,
        market_data=viz_market_data,
        strategy_comparison={"MPC": viz_backtest_result},
        current_weights=curr_w,
    )
    kpis = ts_mpo.get_kpis()
    assert "Immediate Step Turnover" in kpis
    assert "Current Bures-W₂" in kpis
    assert isinstance(ts_mpo.to_figure(), go.Figure)

    html_path = tmp_path / "multi_period_tearsheet.html"
    ts_mpo.to_html(path=html_path)
    assert html_path.exists()


def test_comparison_and_attribution_tearsheet(
    viz_market_data: MarketData,
    viz_scenario_data: ScenarioMarketData,
    tmp_path: Path,
) -> None:
    pipelines = {
        "EqualWeight": StrategyPipeline(
            allocator=EqualWeightAllocator(tickers=viz_market_data.assets)
        ),
        "HRP": StrategyPipeline(
            allocator=HRPAllocator(tickers=viz_market_data.assets),
            cov_estimator=EmpiricalCovariance(),
        ),
        "RiskBudgeting": StrategyPipeline(
            allocator=RiskBudgetingAllocator(tickers=viz_market_data.assets),
            cov_estimator=EmpiricalCovariance(),
        ),
    }

    suite = BenchmarkSuite(
        pipelines=pipelines,
        splitter=RollingWindowSplitter(lookback_window=40),
        rebalance_freq="ME",
        benchmark="EqualWeight",
        reset_tracker=True,
    )
    comp_res = suite.run(viz_market_data, scenario_data=viz_scenario_data)

    fig_eq_dd = plot_equity_drawdown_overlay(
        comp_res, benchmark=comp_res.results["EqualWeight"]
    )
    assert isinstance(fig_eq_dd, go.Figure)
    assert len(fig_eq_dd.data) >= 6

    fig_fan = plot_multi_strategy_fan_chart(comp_res)
    assert isinstance(fig_fan, go.Figure)
    assert len(fig_fan.data) >= 6

    fig_to = plot_multi_turnover_drift(comp_res, market_data=viz_market_data)
    assert isinstance(fig_to, go.Figure)
    assert len(fig_to.data) >= 6

    fig_attr = plot_attribution_waterfall(comp_res, market_data=viz_market_data)
    assert isinstance(fig_attr, go.Figure)
    assert len(fig_attr.data) >= 4

    ts_comp = ComparisonTearSheet(
        comp_res,
        market_data=viz_market_data,
        scenario_data=viz_scenario_data,
        benchmark=comp_res.results["EqualWeight"],
    )
    kpis = ts_comp.get_kpis()
    assert "Top Deflated Sharpe (DSR)" in kpis
    assert "Trials Accounted (N)" in kpis
    assert isinstance(ts_comp.to_figure(), go.Figure)

    html_path = tmp_path / "comparison_tearsheet.html"
    html_str = ts_comp.to_html(path=html_path)
    assert html_path.exists()
    assert "Multi-Strategy Comparison & Attribution Tear Sheet" in html_str
