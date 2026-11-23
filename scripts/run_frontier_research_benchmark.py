"""
Frontier Quantitative Research (2024–2026) Multi-Regime Walk-Forward Benchmark Suite
for PyAlloq.

Spans a multi-regime 2018–2025 timeline (or live YahooFinance data with `--live-yahoo`)
explicitly modeling:
  1. 2018–2019 low-volatility expansion,
  2. 2020 COVID liquidity & volatility shock,
  3. 2021 stimulus recovery,
  4. 2022 inflation & stock-bond correlation breakdown,
  5. 2023–2025 narrow-breadth AI/tech rally.

Compares:
  - Baselines: `HRPAllocator`, `NCOAllocator`, `SetTransformerCovarianceNet` + `MarkowitzAllocator`,
    and `SetTransformerPolicyNet` (`DeepAllocator`).
  - Phase 1 Regime Ablation: `ContinuousJumpModelDetector` (`RegimeSwitchingAllocator` + NCO)
    vs. unconstrained `RollingGaussianHMMDetector` (`mode="partitioned_hmm"`).
  - Phase 2 Meta-Allocation Ablation: `EnsembleAllocator` combining HRP, MeanCVaR, Markowitz,
    and EqualWeight (`1/N` anchor) via `FixedShareStrategy` vs. `DualScaledSwitchingCostStrategy`
    vs. `EllipticalShrinkageStrategy`.
  - Phase 3 & 4 Showcase: `ConformalSafetyGateAllocator` + `ConformalKellyOverlay` and
    Reference-Regulated `MultiPeriodMPCAllocator(reference_allocator=HRPAllocator())`.

Evaluated strictly on excess returns with Location-Scale DSR, MinTRL, Bayesian FDR,
and Top-2 Polyhedral Sharpe Margin Diagnostics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest import (
    AlmgrenChrissCostModel,
    BacktestTracker,
    MultiStrategyComparator,
    RollingWindowSplitter,
)
from pyalloq_core import (
    DualScaledSwitchingCostStrategy,
    EllipticalShrinkageStrategy,
    FixedShareStrategy,
    MarketData,
    StrategyPipeline,
)
from pyalloq_features.regimes import ContinuousJumpModelDetector

from pyalloq.classical.estimators import (
    LedoitWolfShrinkage,
    RelevanceConditionedCovarianceEstimator,
    RelevanceConditionedReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers import (
    ConformalKellyOverlay,
    ConformalSafetyGateAllocator,
    EnsembleAllocator,
    EqualWeightAllocator,
    MarkowitzAllocator,
    MeanCVaRAllocator,
    MultiPeriodMPCAllocator,
    RegimeSwitchingAllocator,
    RiskCertificate,
    RiskParityAllocator,
    RollingGaussianHMMDetector,
)
from pyalloq.dl import (
    DeepAllocator,
    DeepCovarianceEstimator,
    RelationalGraphPriorModule,
    SetTransformerCovarianceNet,
    SetTransformerPolicyNet,
)
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.ml.optimizers.nco import NCOAllocator

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "EEM"]


def build_multi_regime_2018_2025_market_data(
    seed: int = 2026,
    freq: str = "W-FRI",
) -> MarketData:
    """
    Synthesizes a deterministic 2018–2025 multi-asset trajectory capturing:
      - 2018–2019: Low-vol growth (negative stock-bond correlation)
      - 2020 Q1: COVID liquidity crash (all risky assets drop, gold/bonds spike then whip)
      - 2020 Q2–2021: Stimulus bull rally
      - 2022: Inflation shock & stock-bond correlation breakdown (both equities & TLT fall)
      - 2023–2025: Narrow-breadth QQQ/SPY rally with elevated rates
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-05", "2025-08-29", freq=freq)
    n_steps = len(dates)
    n_assets = len(TICKERS)

    scale = np.sqrt(1.0 / 52.0) if "W" in freq else np.sqrt(1.0 / 252.0)
    dt = (1.0 / 52.0) if "W" in freq else (1.0 / 252.0)

    returns = np.zeros((n_steps, n_assets), dtype=np.float64)
    rf_series = np.full(n_steps, 0.02, dtype=np.float64)

    for t, ts in enumerate(dates):
        yr = ts.year
        mo = ts.month
        if yr == 2020 and mo in (2, 3, 4):
            # 2020 Liquidity Shock
            mu_ann = np.array([-0.45, -0.35, 0.25, 0.18, -0.55, -0.48])
            vols_ann = np.array([0.38, 0.40, 0.22, 0.20, 0.45, 0.42])
            corr = np.full((n_assets, n_assets), 0.70)
            corr[2, :] = corr[:, 2] = -0.35
            corr[3, :] = corr[:, 3] = 0.05
            rf_series[t] = 0.005
        elif yr == 2022:
            # 2022 Stock-Bond Correlation Breakdown
            mu_ann = np.array([-0.18, -0.28, -0.24, 0.02, -0.22, -0.20])
            vols_ann = np.array([0.24, 0.30, 0.20, 0.16, 0.26, 0.25])
            corr = np.full((n_assets, n_assets), 0.65)
            rf_series[t] = 0.035
        elif yr >= 2023:
            # 2023-2025 Narrow-Breadth Rally
            mu_ann = np.array([0.18, 0.28, -0.02, 0.14, 0.05, 0.07])
            vols_ann = np.array([0.14, 0.18, 0.16, 0.13, 0.18, 0.17])
            corr = np.full((n_assets, n_assets), 0.30)
            corr[2, :] = corr[:, 2] = 0.15
            rf_series[t] = 0.045
        else:
            # 2018-2019 & 2020H2-2021 Normal Bull
            mu_ann = np.array([0.14, 0.20, 0.05, 0.08, 0.10, 0.09])
            vols_ann = np.array([0.15, 0.18, 0.12, 0.13, 0.17, 0.19])
            corr = np.full((n_assets, n_assets), 0.25)
            corr[2, :] = corr[:, 2] = -0.30
            rf_series[t] = 0.015

        np.fill_diagonal(corr, 1.0)
        eigvals, eigvecs = np.linalg.eigh(corr)
        corr_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-4)) @ eigvecs.T
        cov_step = np.outer(vols_ann * scale, vols_ann * scale) * corr_psd
        chol = np.linalg.cholesky(cov_step + 1e-8 * np.eye(n_assets))
        # Fat-tailed Student-t shocks (df=5)
        z_t = rng.standard_t(df=5, size=n_assets) / np.sqrt(5.0 / 3.0)
        returns[t] = mu_ann * dt + chol @ z_t

    prices_df = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=TICKERS,
    )
    volume_df = pd.DataFrame(
        rng.uniform(30_000_000.0, 150_000_000.0, size=(n_steps, n_assets)),
        index=dates,
        columns=TICKERS,
    )
    cross_sectional = pd.DataFrame(
        {
            "sector": [
                "Equities",
                "Technology",
                "Fixed Income",
                "Commodities",
                "Real Estate",
                "Emerging Markets",
            ],
            "market_cap": [5.0e11, 3.0e11, 6.0e10, 7.0e10, 4.0e10, 3.5e10],
        },
        index=TICKERS,
    )
    return MarketData(
        prices=prices_df,
        features={"volume": volume_df},
        cross_sectional=cross_sectional,
        risk_free_rate=pd.Series(rf_series, index=dates, name="risk_free_rate"),
    )


def run_benchmark(use_live_yahoo: bool = False) -> pd.DataFrame:
    torch.manual_seed(42)
    if use_live_yahoo:
        import datetime as dt

        from pyalloq_data_connector.yahoo_finance import YahooFinanceClient

        connector = YahooFinanceClient()
        data = connector.fetch_historical_prices(
            tickers=TICKERS,
            start_date=dt.datetime(2018, 1, 1, tzinfo=dt.UTC),
            end_date=dt.datetime(2025, 8, 31, tzinfo=dt.UTC),
        )
        data.risk_free_rate = 0.03
    else:
        data = build_multi_regime_2018_2025_market_data()

    ewma_ret = EWMAReturnEstimator(span=20)
    lw_cov = LedoitWolfShrinkage()
    rel_cov = RelevanceConditionedCovarianceEstimator(subsample_fraction=0.35)
    rel_ret = RelevanceConditionedReturnEstimator(subsample_fraction=0.35)

    # Set Transformer Deep Covariance + Relational Graph Prior
    graph_prior = RelationalGraphPriorModule(hidden_dim=8)
    st_cov_net = SetTransformerCovarianceNet(
        n_features=2, hidden_dim=16, num_heads=2, num_inds=4, num_layers=1
    )
    deep_cov_est = DeepCovarianceEstimator(
        model=st_cov_net, lookback_window=20, graph_prior=graph_prior
    )

    st_policy_net = SetTransformerPolicyNet(
        n_features=2,
        hidden_dim=16,
        num_heads=2,
        num_inds=4,
        num_layers=1,
        mandate="capped_simplex",
        max_weight=0.40,
    )
    deep_policy_alloc = DeepAllocator(
        model=st_policy_net, lookback_window=20, graph_prior=graph_prior
    )

    pipelines = {
        "1_Standard_NCO": StrategyPipeline(
            allocator=NCOAllocator(n_clusters=2),
            returns_estimator=ewma_ret,
            cov_estimator=lw_cov,
        ),
        "2_Unconstrained_HMM_NCO": StrategyPipeline(
            allocator=RegimeSwitchingAllocator(
                regime_allocators=[
                    NCOAllocator(n_clusters=2),
                    NCOAllocator(n_clusters=2),
                    NCOAllocator(n_clusters=2),
                ],
                regime_detector=RollingGaussianHMMDetector(
                    n_regimes=3, hard_viterbi=True
                ),
                mode="partitioned_hmm",
            ),
            returns_estimator=ewma_ret,
            cov_estimator=lw_cov,
        ),
        "3_Phase1_CJM_Regime_NCO": StrategyPipeline(
            allocator=RegimeSwitchingAllocator(
                regime_allocators=[
                    NCOAllocator(n_clusters=2),
                    HRPAllocator(),
                    RiskParityAllocator(),
                ],
                regime_detector=ContinuousJumpModelDetector(
                    n_regimes=3, jump_penalty=25.0, mode_loss_weight=0.35
                ),
                mode="probability_blend",
                bear_defensive_cash_weight=0.25,
            ),
            returns_estimator=rel_ret,
            cov_estimator=rel_cov,
        ),
        "4_Phase2_Ensemble_FixedShare": StrategyPipeline(
            allocator=EnsembleAllocator(
                child_allocators=[
                    HRPAllocator(),
                    MeanCVaRAllocator(),
                    MarkowitzAllocator(),
                    EqualWeightAllocator(),
                ],
                meta_strategy=FixedShareStrategy(switching_rate=0.08),
                include_1n_anchor=True,
            ),
            returns_estimator=rel_ret,
            cov_estimator=rel_cov,
        ),
        "5_Phase2_Ensemble_DualScaledOLO": StrategyPipeline(
            allocator=EnsembleAllocator(
                child_allocators=[
                    HRPAllocator(),
                    MeanCVaRAllocator(),
                    MarkowitzAllocator(),
                    EqualWeightAllocator(),
                ],
                meta_strategy=DualScaledSwitchingCostStrategy(
                    switching_cost_penalty=0.35
                ),
                include_1n_anchor=True,
            ),
            returns_estimator=rel_ret,
            cov_estimator=rel_cov,
        ),
        "6_Phase2_Ensemble_Elliptical": StrategyPipeline(
            allocator=EnsembleAllocator(
                child_allocators=[
                    MarkowitzAllocator(),
                    HRPAllocator(),
                    EqualWeightAllocator(),
                ],
                meta_strategy=EllipticalShrinkageStrategy(risk_aversion=2.5),
                include_1n_anchor=True,
            ),
            returns_estimator=ewma_ret,
            cov_estimator=lw_cov,
        ),
        "7_Phase3_ConformalSafetyGate_SetPolicy": StrategyPipeline(
            allocator=ConformalSafetyGateAllocator(
                base_allocator=ConformalKellyOverlay(
                    base_allocator=deep_policy_alloc,
                    alpha=0.25,
                    kelly_fraction=0.65,
                ),
                certificate=RiskCertificate(
                    max_tail_loss=0.035,
                    max_gross_leverage=1.0,
                    max_turnover=0.35,
                    max_annual_volatility=0.22,
                ),
            ),
            returns_estimator=ewma_ret,
            cov_estimator=deep_cov_est,
        ),
        "8_Phase4_RefRegulated_MPC": StrategyPipeline(
            allocator=MultiPeriodMPCAllocator(
                horizon=4,
                reference_allocator=HRPAllocator(),
                reference_penalty=1.5,
                backend="admm",
            ),
            returns_estimator=rel_ret,
            cov_estimator=rel_cov,
        ),
    }

    with BacktestTracker.session(reset=True):
        comparator = MultiStrategyComparator(
            pipelines=pipelines,
            splitter=RollingWindowSplitter(lookback_window=52),
            cost_model=AlmgrenChrissCostModel(
                portfolio_aum=10_000_000.0, spread_bps=10.0, gamma=0.05
            ),
            rebalance_freq="ME",
            rebalance_threshold=0.05,
            max_workers=1,
        )
        comp_res = comparator.run(data)

    cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "location_scale_dsr",
        "min_track_record_length",
        "bayesian_fdr",
        "annualized_turnover",
        "total_cost_drag",
    ]
    present_cols = [c for c in cols if c in comp_res.summary_table.columns]
    df_out = comp_res.summary_table[present_cols]

    print("=" * 105)
    print("PyAlloq 2024–2026 Frontier Research Walk-Forward Benchmark (2018–2025)")
    print("=" * 105)
    print(df_out.round(4).to_string())
    print("-" * 105)
    print(
        f"Trials Tracked: {comp_res.n_trials} | Expected Max Sharpe (SR0): {comp_res.expected_max_sharpe:.4f} | "
        f"Top-2 Sharpe Gap: {comp_res.top2_sharpe_gap:.4f} (Warning={comp_res.top2_sharpe_gap_warning}) | "
        f"Strict Excess Returns: {comp_res.used_excess_returns}"
    )
    print("=" * 105)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyAlloq 2024-2026 Frontier Research Benchmark"
    )
    parser.add_argument(
        "--live-yahoo",
        action="store_true",
        help="Fetch live 2018-2025 prices via YahooFinanceConnector instead of offline multi-regime generator",
    )
    args = parser.parse_args()
    run_benchmark(use_live_yahoo=args.live_yahoo)


if __name__ == "__main__":
    main()
