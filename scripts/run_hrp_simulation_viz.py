"""
End-to-end example:
1. Fetch historical data for 10 randomly selected stocks (using Yahoo Finance with offline fallback).
2. Optimize portfolio weights with Hierarchical Risk Parity (HRP) and run a historical walk-forward backtest.
3. Simulate synthetic multiverse scenarios using GMM Markov Regime Generator and apply Meucci Entropy Pooling views.
4. Generate and save all 4 interactive HTML visualization tear sheets:
   - Simulation & Multiverse Tear Sheet
   - Robustness & Overfitting (PBO / CPCV) Tear Sheet
   - Macro Regime & Entropy Tear Sheet
   - Allocation Mechanics & Risk Tear Sheet
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyalloq_viz as viz
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.overfitting import CPCVSplitter
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient
from pyalloq_simulation import (
    AbsoluteView,
    EntropyPooler,
    GMMRegimeGenerator,
    ViewBuilder,
)

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.ml.optimizers.hrp import HRPAllocator

# Candidate stock universe across multiple sectors
STOCK_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",  # Tech
    "JPM",
    "BAC",
    "GS",  # Financials
    "JNJ",
    "PFE",
    "UNH",  # Healthcare
    "XOM",
    "CVX",  # Energy
    "PG",
    "KO",
    "PEP",  # Consumer Staples
    "TSLA",
    "HD",
    "MCD",  # Consumer Discretionary
    "CAT",
    "BA",  # Industrials
]

SECTOR_MAP = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "JNJ": "Health Care",
    "PFE": "Health Care",
    "UNH": "Health Care",
    "XOM": "Energy",
    "CVX": "Energy",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "CAT": "Industrials",
    "BA": "Industrials",
}


def get_stock_data(
    tickers: list[str], start_date: dt.datetime, end_date: dt.datetime
) -> MarketData:
    """Fetch live data from Yahoo Finance, falling back to realistic synthetic data if offline."""
    print(
        f"Attempting to fetch data for {len(tickers)} tickers via Yahoo Finance: {tickers}..."
    )
    try:
        client = YahooFinanceClient()
        market_data = client.get_market_data(tickers, start=start_date, end=end_date)
        # Verify valid non-empty prices with at least 100 rows
        if market_data.prices is not None and len(market_data.prices) > 100:
            print(
                f"Successfully fetched {len(market_data.prices)} daily observations from Yahoo Finance."
            )
            # Attach sector metadata for hierarchical treemaps
            cross_sec = pd.DataFrame(
                {"Sector": [SECTOR_MAP.get(t, "Other") for t in market_data.assets]},
                index=market_data.assets,
            )
            market_data.cross_sectional = cross_sec
            return market_data
    except Exception as exc:
        print(
            f"Note: Yahoo Finance fetch not available ({exc}). Falling back to synthetic market data."
        )

    # Offline realistic fallback
    print("Generating synthetic market data for demonstration...")
    np.random.seed(42)
    dates = pd.bdate_range(start_date, end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    # Correlated asset returns with sector-like structure
    base_factor = np.random.normal(0.0004, 0.012, size=n_days)
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        idio = np.random.normal(0.0002, 0.015, size=n_days)
        returns[:, i] = 0.5 * base_factor + 0.5 * idio

    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=tickers,
    )
    cross_sec = pd.DataFrame(
        {"Sector": [SECTOR_MAP.get(t, "Other") for t in tickers]},
        index=tickers,
    )
    return MarketData(prices=prices, cross_sectional=cross_sec, risk_free_rate=0.03)


def main():
    output_dir = Path("visualizations_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select 10 random stocks from the universe
    random.seed(42)  # reproducible random choice
    selected_tickers = sorted(random.sample(STOCK_UNIVERSE, 10))
    print(f"Selected 10 random stocks: {selected_tickers}")

    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2024, 1, 1)
    market_data = get_stock_data(
        selected_tickers, start_date=start_date, end_date=end_date
    )

    # 2. Hierarchical Risk Parity (HRP) Allocation & Historical Walk-Forward Backtest
    print("\n--- 1. Running Hierarchical Risk Parity (HRP) Optimization & Backtest ---")
    cov_estimator = EmpiricalCovariance(annualization_factor=252.0)
    allocator = HRPAllocator(tickers=market_data.assets)
    pipeline = StrategyPipeline(allocator=allocator, cov_estimator=cov_estimator)

    engine = WalkForwardEngine(
        pipeline=pipeline,
        splitter=RollingWindowSplitter(lookback_window=126),  # 6-month lookback
        rebalance_freq="ME",  # Monthly rebalancing
    )
    backtest_result = engine.run(market_data)
    print("Backtest Complete! Historical Performance Summary:")
    print(backtest_result.tear_sheet)

    # Current optimal weights from HRP
    cov_matrix = market_data.get_returns().cov() * 252.0
    exp_returns = market_data.get_returns().mean() * 252.0
    hrp_result = allocator.allocate(market_data, cov_matrix=cov_matrix)
    print("\nCurrent HRP Portfolio Weights:")
    for ticker, weight in hrp_result.clean_weights().items():
        if weight > 0:
            print(f"  {ticker:6s}: {weight:6.2%}")

    # 3. Simulate Synthetic Scenarios (GMM Markov Regime-Switching Multiverse)
    print("\n--- 2. Fitting GMM Markov Regime Generator & Simulating 500 Paths ---")
    gmm_gen = GMMRegimeGenerator(n_regimes=2, random_state=42)
    gmm_gen.fit(market_data)

    # Generate 500 future paths across a 63-day horizon (quarterly)
    scenarios = gmm_gen.generate(n_paths=500, horizon=63)
    print(
        f"Simulated Scenario Tensor shape: {scenarios.prices.shape} (paths, days, assets)"
    )

    # 4. Express Subjective Views via Meucci's Entropy Pooling
    print("\n--- 3. Formulating Views & Running Entropy Pooling ---")
    # View: Expect first asset to return >= +3% over horizon
    target_asset = market_data.assets[0]
    builder = ViewBuilder(scenarios)
    A_ineq, b_ineq, C_eq, d_eq = builder.build(
        [AbsoluteView(asset=target_asset, operator=">=", target=0.03)]
    )
    pooler = EntropyPooler(n_scenarios=scenarios.n_paths)
    entropy_result = pooler.fit(A_ineq=A_ineq, b_ineq=b_ineq, C_eq=C_eq, d_eq=d_eq)
    print(
        f"Entropy Pooling Converged: {entropy_result.success} | KL Divergence: {entropy_result.kl_divergence:.4f}"
    )

    # 5. Combinatorial Purged Cross-Validation (CPCV) Scrambling
    print("\n--- 4. Generating Combinatorial Regime-Scrambled (CPCV) Timelines ---")
    cpcv_splitter = CPCVSplitter(
        n_groups=6, k_test_groups=2, max_paths=60, random_state=42
    )
    cpcv_result = cpcv_splitter.scramble_timelines(backtest_result)
    print(
        f"Generated {cpcv_result.equity_curves.shape[1]} scrambled timeline paths across 6 regime blocks."
    )

    # 6. Render & Export All 4 Interactive HTML Tear Sheets
    print("\n--- 5. Exporting Interactive Visualization Tear Sheets ---")

    # Tear Sheet 1: The Multiverse & Simulation Tear Sheet
    sim_html = output_dir / "1_simulation_tearsheet.html"
    sim_ts = viz.SimulationTearSheet(
        scenarios,
        asset_weights=hrp_result.weights,
        entropy_result=entropy_result,
        ruin_threshold=-0.15,
        cvar_alpha=0.05,
        benchmark=backtest_result,
    )
    sim_ts.to_html(sim_html, theme="alloq_dark")
    print(f"  [✓] Simulation Tear Sheet saved to: {sim_html}")

    # Tear Sheet 2: Robustness & Overfitting Tear Sheet
    rob_html = output_dir / "2_robustness_tearsheet.html"
    rob_ts = viz.RobustnessTearSheet(backtest_result, cpcv_or_synthetic=cpcv_result)
    rob_ts.to_html(rob_html, theme="alloq_dark")
    print(f"  [✓] Robustness Tear Sheet saved to: {rob_html}")

    # Tear Sheet 3: Macro Regime & Entropy Tear Sheet
    macro_html = output_dir / "3_macro_regime_tearsheet.html"
    macro_ts = viz.MacroRegimeTearSheet(
        entropy_result=entropy_result,
        equity_or_scenarios=backtest_result,
        gmm_or_probs=gmm_gen,
        scenario_data=scenarios,
        market_data=market_data,
        asset_weights=hrp_result.weights,
    )
    macro_ts.to_html(macro_html, theme="alloq_dark")
    print(f"  [✓] Macro Regime Tear Sheet saved to: {macro_html}")

    # Tear Sheet 4: Allocation Mechanics & Risk Tear Sheet
    alloc_html = output_dir / "4_allocation_tearsheet.html"
    alloc_ts = viz.AllocationTearSheet(
        optimization_result=hrp_result,
        backtest_result=backtest_result,
        market_data=market_data,
        cov_matrix=cov_matrix,
        expected_returns=exp_returns,
        scenario_data=scenarios,
        allocator_or_pipeline=allocator,
    )
    alloc_ts.to_html(alloc_html, theme="alloq_dark")
    print(f"  [✓] Allocation Mechanics Tear Sheet saved to: {alloc_html}")

    print("\n" + "=" * 70)
    print(f"All visualizations exported successfully into '{output_dir.resolve()}/'!")
    print(
        "Open any of the generated HTML files in your browser to inspect interactive charts."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
