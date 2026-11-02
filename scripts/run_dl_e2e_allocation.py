"""
End-to-End Deep Learning (DL E2E) Portfolio Allocation Example:
1. Loads historical market data (via Yahoo Finance with deterministic synthetic fallback)
   and computes rolling momentum & volatility features.
2. Instantiates the ConstrainedSpatioTemporalAttentionNet (C-STAN) neural allocator:
   - AssetTemporalEncoder (LSTM per asset)
   - PortfolioSpatialAttention (Multi-Head Cross-Asset Attention replacing explicit covariance)
   - AssetScoringLayer + BoundedSoftmax (differentiable water-filling weight cap)
3. Trains the network using `WalkForwardTrainer` and transaction-cost-aware `SharpeLoss`.
4. Evaluates the trained `DeepAllocator` inside `WalkForwardEngine` (with intra-period
   weight drift, threshold rebalancing, and benchmark comparison against Equal-Weight and HRP).
5. Exports interactive HTML Tear Sheets (`AllocationTearSheet` & `RobustnessTearSheet`).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyalloq_viz as viz
import torch
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.overfitting import CPCVSplitter
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_data_connector.yahoo_finance import YahooFinanceClient

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq.dl.estimators.e2e.deep_allocator import DeepAllocator
from pyalloq.dl.estimators.e2e.models.c_stan import (
    ConstrainedSpatioTemporalAttentionNet,
)
from pyalloq.dl.loss_functions.sharpe import SharpeLoss
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.ml.optimizers.hrp import HRPAllocator

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "UNH", "XOM", "PG", "CAT"]

SECTOR_MAP = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "JPM": "Financials",
    "UNH": "Health Care",
    "XOM": "Energy",
    "PG": "Consumer Staples",
    "CAT": "Industrials",
}


def build_market_data(
    tickers: list[str], start_date: dt.datetime, end_date: dt.datetime
) -> MarketData:
    """Fetch market prices (or generate synthetic multi-factor prices offline) and attach features."""
    prices: pd.DataFrame | None = None
    try:
        client = YahooFinanceClient()
        md = client.get_market_data(tickers, start=start_date, end=end_date)
        if md.prices is not None and len(md.prices) > 150:
            prices = md.prices
            print(
                f"Fetched {len(prices)} daily rows for {len(tickers)} assets from Yahoo Finance."
            )
    except Exception as exc:
        print(
            f"Yahoo Finance unavailable ({exc}); using synthetic multi-factor market data."
        )

    if prices is None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start_date, end_date)
        n_days = len(dates)
        n_assets = len(tickers)

        # Create persistent momentum & volatility regimes so C-STAN has learnable spatiotemporal structure
        market_factor = rng.normal(0.0004, 0.010, size=n_days)
        tech_factor = np.sin(np.linspace(0, 6 * np.pi, n_days)) * 0.002
        returns = np.zeros((n_days, n_assets))
        for i in range(n_assets):
            beta_mkt = 0.6 + 0.1 * (i % 3)
            beta_tech = 1.0 if i < 3 else -0.3
            idio = rng.normal(0.0001, 0.011, size=n_days)
            returns[:, i] = beta_mkt * market_factor + beta_tech * tech_factor + idio

        prices = pd.DataFrame(
            100.0 * np.cumprod(1.0 + returns, axis=0),
            index=dates,
            columns=tickers,
        )

    # Construct point-in-time rolling features (20d momentum & 20d realized volatility)
    daily_ret = prices.pct_change().fillna(0.0)
    mom_20d = prices.pct_change(20).fillna(0.0)
    vol_20d = daily_ret.rolling(20).std().bfill().fillna(0.01)

    cross_sec = pd.DataFrame(
        {"Sector": [SECTOR_MAP.get(t, "Other") for t in tickers]},
        index=tickers,
    )

    return MarketData(
        prices=prices,
        features={
            "momentum_20d": mom_20d,
            "volatility_20d": vol_20d,
        },
        cross_sectional=cross_sec,
        risk_free_rate=0.03,
    )


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = Path("visualizations_output") / "dl_e2e"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = dt.datetime(2022, 1, 3)
    end_date = dt.datetime(2024, 6, 28)
    market_data = build_market_data(TICKERS, start_date=start_date, end_date=end_date)

    # 1 base return channel + number of feature DataFrames in market_data.features
    n_features = 1 + len(market_data.features)
    lookback_window = 30
    max_asset_weight = 0.30

    print("\n--- 1. Initializing C-STAN End-to-End Deep Portfolio Allocator ---")
    cstan_model = ConstrainedSpatioTemporalAttentionNet(
        n_features=n_features,
        hidden_dim=32,
        num_heads=4,
        dropout=0.1,
        max_asset_weight=max_asset_weight,
        waterfill_iterations=4,
    )

    cost_model = FlatBpsCostModel(bps=10.0)
    loss_fn = SharpeLoss(cost_model=cost_model, risk_free_rate=0.03 / 252.0)
    optimizer = torch.optim.AdamW(cstan_model.parameters(), lr=3e-3, weight_decay=1e-4)

    train_splitter = RollingWindowSplitter(lookback_window=126)
    idx_series = pd.Series(market_data.prices.index, index=market_data.prices.index)
    rebalance_dates = pd.DatetimeIndex(
        idx_series.resample("QE").last().dropna().values
    ).unique()

    print("\n--- 2. Walk-Forward Training C-STAN with Net-of-Cost SharpeLoss ---")
    trainer = WalkForwardTrainer(
        model=cstan_model,
        loss_function=loss_fn,
        optimizer=optimizer,
        splitter=train_splitter,
        lookback_window=lookback_window,
        horizon=1,
        batch_size=16,
        epochs_per_window=5,
        device="cpu",
    )
    train_history = trainer.train(
        data=market_data,
        rebalance_dates=rebalance_dates,
        reset_weights_each_window=False,
    )
    print(
        f"Completed {len(train_history['window_date'])} walk-forward training windows."
    )

    # 3. Benchmark Strategies (Equal Weight & HRP) vs. Trained DL E2E Allocator
    print("\n--- 3. Running Walk-Forward Backtest: C-STAN DL E2E vs. Benchmarks ---")
    eval_splitter = RollingWindowSplitter(lookback_window=126)

    # Equal-Weight Benchmark
    ew_pipe = StrategyPipeline(
        allocator=EqualWeightAllocator(tickers=market_data.assets)
    )
    ew_engine = WalkForwardEngine(
        pipeline=ew_pipe,
        splitter=eval_splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
    )
    ew_bt = ew_engine.run(market_data)

    # HRP Benchmark
    hrp_allocator = HRPAllocator(tickers=market_data.assets)
    hrp_pipe = StrategyPipeline(
        allocator=hrp_allocator,
        cov_estimator=EmpiricalCovariance(annualization_factor=252.0),
    )
    hrp_engine = WalkForwardEngine(
        pipeline=hrp_pipe,
        splitter=eval_splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
        benchmark_returns=ew_bt.returns,
    )
    hrp_bt = hrp_engine.run(market_data)

    # DeepAllocator (C-STAN E2E) with threshold-aware drifted rebalancing
    deep_allocator = DeepAllocator(
        model=cstan_model,
        lookback_window=lookback_window,
        device="cpu",
        tickers=market_data.assets,
    )
    dl_pipe = StrategyPipeline(allocator=deep_allocator)
    dl_engine = WalkForwardEngine(
        pipeline=dl_pipe,
        splitter=eval_splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
        drift_weights=True,
        rebalance_threshold=0.02,  # Skip noisy micro-rebalances under 2% L1 drift
        benchmark_returns=ew_bt.returns,
    )
    dl_bt = dl_engine.run(market_data)

    comparison = pd.DataFrame(
        {
            "EqualWeight": ew_bt.tear_sheet["Value"],
            "HRP": hrp_bt.tear_sheet["Value"],
            "DL_E2E_CSTAN": dl_bt.tear_sheet["Value"],
        }
    )
    print("\n=== Strategy Tear Sheet Comparison ===")
    print(comparison.round(4))

    # Latest point-in-time allocation from DeepAllocator
    latest_opt = deep_allocator.allocate(market_data)
    print(
        f"\nLatest C-STAN Optimal Allocation (max weight cap = {max_asset_weight:.0%}):"
    )
    for ticker, w in latest_opt.clean_weights().items():
        print(f"  {ticker:6s}: {w:6.2%}")

    # 4. Generate Interactive Visualization Tear Sheets
    print("\n--- 4. Exporting Interactive Visualization Tear Sheets ---")
    cov_matrix = market_data.get_returns().cov() * 252.0
    exp_returns = market_data.get_returns().mean() * 252.0

    cpcv = CPCVSplitter(n_groups=6, k_test_groups=2, max_paths=50, random_state=42)
    cpcv_res = cpcv.scramble_timelines(dl_bt)

    rob_html = output_dir / "1_dl_e2e_robustness_tearsheet.html"
    rob_ts = viz.RobustnessTearSheet(dl_bt, cpcv_or_synthetic=cpcv_res)
    rob_ts.to_html(rob_html, theme="alloq_dark")
    print(f"  [✓] DL E2E Robustness Tear Sheet saved to: {rob_html}")

    alloc_html = output_dir / "2_dl_e2e_allocation_tearsheet.html"
    alloc_ts = viz.AllocationTearSheet(
        optimization_result=latest_opt,
        backtest_result=dl_bt,
        market_data=market_data,
        cov_matrix=cov_matrix,
        expected_returns=exp_returns,
    )
    alloc_ts.to_html(alloc_html, theme="alloq_dark")
    print(f"  [✓] DL E2E Allocation Tear Sheet saved to: {alloc_html}")


if __name__ == "__main__":
    main()
