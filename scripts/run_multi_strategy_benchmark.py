"""
End-to-End Multi-Strategy Benchmark Suite, Deflated Sharpe Ratio (DSR),
Brinson-Fachler / Factor / Frictional Attribution & ComparisonTearSheet Script.

Demonstrates:
1. `BacktestTracker` singleton automatically tracking trial count N and cross-trial Sharpe variance.
2. `MultiStrategyComparator` (`BenchmarkSuite`) running `EqualWeightAllocator`, `RiskParityAllocator`,
   `HRPAllocator`, `MarkowitzAllocator`, and `MultiPeriodMPCAllocator` concurrently over identical
   `MarketData` under `AlmgrenChrissCostModel` (proportional spread + square-root market impact).
3. `PerformanceAttributor` decomposing active return into:
   - Classical Brinson-Fachler (`Allocation`, `Selection`, `Interaction` with Carino linking)
   - Factor-Implied Attribution (OLS + Newey-West HAC exposures to Market, Momentum, LowVolatility)
   - Frictional Drag Accounting (Ma & Smith, 2025: Paper Return -> Alpha Decay/Drift -> Spread -> Impact -> Net)
4. `ComparisonTearSheet` exporting interactive trajectory overlays, fan cones, sawtooth drift,
   and attribution waterfall to HTML.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyalloq_backtest import (
    AlmgrenChrissCostModel,
    BacktestTracker,
    BenchmarkSuite,
    RollingWindowSplitter,
)
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_simulation import GMMRegimeGenerator
from pyalloq_viz import ComparisonTearSheet

from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.multi_period import MultiPeriodMPCAllocator
from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator
from pyalloq.ml.optimizers.hrp import HRPAllocator

TICKERS = ["AAPL", "MSFT", "NVDA", "TLT", "GLD", "XLE"]


def build_benchmark_market_data(
    tickers: list[str], n_days: int = 320, seed: int = 42
) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    n_assets = len(tickers)

    base_corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(base_corr, 1.0)
    vols = np.array([0.015, 0.014, 0.022, 0.008, 0.009, 0.016])
    cov = np.outer(vols, vols) * base_corr
    chol = np.linalg.cholesky(cov)

    drifts = np.array([0.0005, 0.0005, 0.0008, 0.0002, 0.0003, 0.0004])
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    returns = drifts + shocks

    prices_df = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers
    )
    volume_df = pd.DataFrame(
        rng.uniform(25_000_000.0, 120_000_000.0, size=(n_days, n_assets)),
        index=dates,
        columns=tickers,
    )
    cross_sectional = pd.DataFrame(
        {
            "sector": [
                "Technology",
                "Technology",
                "Technology",
                "Fixed Income",
                "Commodities",
                "Energy",
            ],
            "market_cap": [3.0e12, 2.9e12, 2.5e12, 5.0e10, 6.0e10, 4.0e11],
            "book_to_market": [0.15, 0.18, 0.10, 0.85, 0.70, 0.65],
        },
        index=tickers,
    )

    return MarketData(
        prices=prices_df,
        features={"volume": volume_df},
        cross_sectional=cross_sectional,
        risk_free_rate=0.03,
    )


def main() -> None:
    print("=" * 88)
    print(
        "PyAlloq Multi-Strategy Benchmark Suite (DSR, Brinson-Fachler, Factor & Frictional Drag)"
    )
    print("=" * 88)

    market_data = build_benchmark_market_data(TICKERS, n_days=300)
    cov_est = LedoitWolfShrinkage()

    # Simulate 80 multi-period synthetic futures for the Cone & Fan Chart comparison
    gmm = GMMRegimeGenerator(n_regimes=2, random_state=42)
    gmm.fit(market_data)
    scenarios = gmm.generate(n_paths=80, horizon=45)

    pipelines = {
        "EqualWeight": StrategyPipeline(
            allocator=EqualWeightAllocator(tickers=TICKERS)
        ),
        "RiskParity": StrategyPipeline(
            allocator=RiskParityAllocator(tickers=TICKERS),
            cov_estimator=cov_est,
        ),
        "HRP": StrategyPipeline(
            allocator=HRPAllocator(tickers=TICKERS),
            cov_estimator=cov_est,
        ),
        "Markowitz": StrategyPipeline(
            allocator=MarkowitzAllocator(
                tickers=TICKERS, objective=ObjectiveFunction.MAX_RETURN
            ),
            returns_estimator=EWMAReturnEstimator(span=30),
            cov_estimator=cov_est,
        ),
        "MultiPeriodMPC": StrategyPipeline(
            allocator=MultiPeriodMPCAllocator(
                tickers=TICKERS,
                horizon=5,
                risk_aversion=3.0,
                proportional_cost_bps=10.0,
                market_impact_coeff=0.03,
                backend="admm",
            ),
            returns_estimator=AlphaDecayReturnEstimator(horizon=5, alpha_decay=0.15),
            cov_estimator=cov_est,
        ),
    }

    cost_model = AlmgrenChrissCostModel(
        portfolio_aum=10_000_000.0,
        spread_bps=4.0,
        gamma=0.10,
    )

    with BacktestTracker.session(reset=True) as tracker:
        suite = BenchmarkSuite(
            pipelines=pipelines,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=cost_model,
            rebalance_freq="W-FRI",
            benchmark="EqualWeight",
        )
        comp_result = suite.run(market_data, scenario_data=scenarios)

        cols_display = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "probabilistic_sharpe_ratio",
            "deflated_sharpe_ratio",
            "maximum_drawdown",
            "annualized_turnover",
            "total_cost_drag",
        ]
        print(
            f"\nExecuted {tracker.n_trials} Strategy Trials | "
            f"Expected Max Sharpe Hurdle (SR_0): {comp_result.expected_max_sharpe:.3f}"
        )
        print("\n1. Multi-Strategy Summary Table (Raw Metrics + Rigorous DSR):")
        print(comp_result.summary_table[cols_display].round(4).to_string())

        mpc_attr = comp_result.attributions["MultiPeriodMPC"]
        print(
            "\n2. Classical Brinson-Fachler Decomposition (MultiPeriodMPC vs. EqualWeight):"
        )
        print(mpc_attr.brinson.sector_summary.round(4).to_string())

        print("\n3. Factor-Implied Attribution (MultiPeriodMPC OLS + Newey-West HAC):")
        print(mpc_attr.factor.factor_summary.round(4).to_string())

        print("\n4. Frictional Drag Accounting Waterfall (Ma & Smith, 2025):")
        print(mpc_attr.frictional.waterfall_summary.round(4).to_string())

        out_dir = Path("visualizations_output/benchmark")
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "multi_strategy_comparison_tearsheet.html"

        ts = ComparisonTearSheet(
            comp_result,
            market_data=market_data,
            scenario_data=scenarios,
            benchmark=comp_result.results["EqualWeight"],
        )
        ts.to_html(path=html_path)
        print(f"\nSaved interactive ComparisonTearSheet to: {html_path.resolve()}")


if __name__ == "__main__":
    main()
