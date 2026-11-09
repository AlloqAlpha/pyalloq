"""
Point-in-Time (PIT) Option-Implied & Rough Volatility + Graph Reaction-Diffusion Contagion Example:
1. Constructs synthetic multi-sector market data and Point-in-Time (PIT) option chains
   (including post-close stale/future quotes to verify strict PIT filtering).
2. Runs the Feature Engineering pipeline:
   - `PITOptionsMFIVTransformer` (Britten-Jones & Neuberger 2000 Model-Free Implied Volatility)
   - `LiftedHestonRoughVolTransformer` (Abi Jaber 2019 K-factor Markovian rough volatility & adaptive Hurst H_eff)
   - `LogSignatureTransformer` (Lyndon-basis iterated integrals & price-vol Lévy areas)
3. Benchmarks Classical HRP (`HRPAllocator` + `EmpiricalCovariance`) vs.
    Option-Implied HRP (`ImpliedCovarianceEstimator`) vs.
   Graph Contagion Spectral HRP (`ReactionDiffusionCovarianceEstimator` + `SpectralLaplacianHRPAllocator`)
   inside `WalkForwardEngine`.
4. Fits `ReactionDiffusionGenerator` (3D `ScenarioMarketData` tensor) to simulate cross-asset
   volatility cascades from a localized hub shock and exports interactive HTML tear sheets.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyalloq_viz as viz
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_features import (
    LiftedHestonRoughVolTransformer,
    LogSignatureTransformer,
    PITOptionsMFIVTransformer,
)
from pyalloq_simulation import (
    EntropyPooler,
    ReactionDiffusionGenerator,
    ScenarioEvaluator,
)

from pyalloq import (
    EmpiricalCovariance,
    HRPAllocator,
    ImpliedCovarianceEstimator,
    ReactionDiffusionCovarianceEstimator,
    SpectralLaplacianHRPAllocator,
)

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "GS", "XOM"]


def build_synthetic_market_and_pit_options(
    tickers: list[str],
    start_date: dt.datetime,
    end_date: dt.datetime,
) -> tuple[MarketData, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start_date, end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    tech_shock = rng.normal(0.0004, 0.013, size=n_days)
    fin_shock = rng.normal(0.0003, 0.011, size=n_days)
    returns = np.zeros((n_days, n_assets), dtype=np.float64)

    for j in range(n_assets):
        sector_factor = tech_shock if j < 3 else fin_shock
        idio = rng.normal(0.0001, 0.009, size=n_days)
        returns[:, j] = 0.7 * sector_factor + 0.3 * idio

    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=tickers,
    )

    # Generate synthetic PIT option chains for the last 20 trading days
    option_rows: list[dict[str, object]] = []
    for dt_val in dates[-20:]:
        exp_dt = dt_val + pd.Timedelta(days=30)
        for ticker in tickers[:3]:
            spot = float(prices[ticker].loc[dt_val])
            for m in np.linspace(0.85, 1.15, 7):
                strike = round(spot * float(m), 2)
                opt_type = "P" if strike < spot else "C"
                dist = abs(strike - spot) / spot
                mid_price = max(0.25, spot * 0.045 * float(np.exp(-5.0 * dist)))
                option_rows.append(
                    {
                        "date": dt_val,
                        "ticker": ticker,
                        "expiration": exp_dt,
                        "strike": strike,
                        "option_type": opt_type,
                        "bid": mid_price * 0.96,
                        "ask": mid_price * 1.04,
                        "quote_timestamp": dt_val + pd.Timedelta(hours=15, minutes=45),
                    }
                )
            # Add a post-close look-ahead quote at 18:30 that must be filtered out by PIT
            option_rows.append(
                {
                    "date": dt_val,
                    "ticker": ticker,
                    "expiration": exp_dt,
                    "strike": round(spot, 2),
                    "option_type": "C",
                    "bid": spot * 0.50,
                    "ask": spot * 0.55,
                    "quote_timestamp": dt_val + pd.Timedelta(hours=18, minutes=30),
                }
            )

    chains_df = pd.DataFrame(option_rows)
    market_data = MarketData(prices=prices, risk_free_rate=0.03)
    return market_data, chains_df


def main() -> None:
    output_dir = Path("visualizations_output") / "pit_reaction_diffusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data, chains_df = build_synthetic_market_and_pit_options(
        TICKERS,
        start_date=dt.datetime(2023, 1, 2),
        end_date=dt.datetime(2024, 3, 29),
    )

    print(
        "--- 1. Running PIT MFIV, Lifted Heston Rough Vol & Log-Signature Transformers ---"
    )
    market_data = PITOptionsMFIVTransformer(
        option_chain_df=chains_df,
        target_horizon_days=30.0,
        market_close_offset_hours=16.0,
    ).transform(market_data)
    market_data = LiftedHestonRoughVolTransformer(
        n_factors=5,
        estimation_window=25,
    ).transform(market_data)
    market_data = LogSignatureTransformer(window=8, depth=3).transform(market_data)

    print("Extracted PIT Feature Keys:", list(market_data.features.keys()))
    latest_hurst = market_data.features["hurst_exponent"].iloc[-1]
    print("Latest Effective Hurst Exponents (H_eff):")
    for t, h in latest_hurst.items():
        print(f"  {t:6s}: H_eff = {h:.3f}")

    print(
        "\n--- 2. Evaluating Spectral Graph Fragility & Wiener-Hopf Hawkes Criticality ---"
    )
    rd_cov_est = ReactionDiffusionCovarianceEstimator(
        diffusion_coeff=0.35,
        horizon_steps=5,
    )
    cov_rd = rd_cov_est.estimate(market_data)
    diag = rd_cov_est.last_diagnostics_
    if diag is not None:
        print(f"  Fiedler Value (lambda_2):         {diag.fiedler_value:.4f}")
        print(f"  Network Mixing Time (tau_mix):    {diag.mixing_time:.2f} steps")
        print(f"  Hawkes Spectral Radius rho(G):    {diag.hawkes_spectral_radius:.4f}")
        print(
            f"  Fragility Shrinkage Alpha:        {diag.fragility_shrinkage_alpha:.4f}"
        )

    print(
        "\n--- 3. Walk-Forward Backtest: Classical HRP vs. Implied HRP vs. Spectral Laplacian HRP ---"
    )
    splitter = RollingWindowSplitter(lookback_window=100)
    cost_model = FlatBpsCostModel(bps=10.0)

    pipe_classical = StrategyPipeline(
        cov_estimator=EmpiricalCovariance(),
        allocator=HRPAllocator(tickers=TICKERS, clustering_mode="single_linkage"),
    )
    pipe_implied = StrategyPipeline(
        cov_estimator=ImpliedCovarianceEstimator(index_implied_vol=0.22),
        allocator=HRPAllocator(tickers=TICKERS, clustering_mode="single_linkage"),
    )
    pipe_spectral_rd = StrategyPipeline(
        cov_estimator=rd_cov_est,
        allocator=SpectralLaplacianHRPAllocator(
            tickers=TICKERS,
            fragility_penalty=0.25,
        ),
    )

    bt_classical = WalkForwardEngine(
        pipeline=pipe_classical,
        splitter=splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
    ).run(market_data)
    bt_implied = WalkForwardEngine(
        pipeline=pipe_implied,
        splitter=splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
    ).run(market_data)
    bt_spectral = WalkForwardEngine(
        pipeline=pipe_spectral_rd,
        splitter=splitter,
        cost_model=cost_model,
        rebalance_freq="ME",
    ).run(market_data)

    summary_df = pd.DataFrame(
        {
            "Classical_HRP": bt_classical.tear_sheet["Value"],
            "Implied_BussVilkov_HRP": bt_implied.tear_sheet["Value"],
            "ReactionDiffusion_Spectral_HRP": bt_spectral.tear_sheet["Value"],
        }
    )
    print(summary_df.round(4))

    print(
        "\n--- 4. Simulating 3D Contagion Scenarios via ReactionDiffusionGenerator ---"
    )
    rd_gen = ReactionDiffusionGenerator(
        diffusion_coeff=0.35,
        leverage_rho=-0.55,
        random_state=42,
    )
    rd_gen.fit(market_data)
    scenarios = rd_gen.generate(
        n_paths=150,
        horizon=30,
        initial_shock={"NVDA": 0.75},
    )
    print(f"Generated ScenarioMarketData Tensor: prices={scenarios.prices.shape}")

    spec_res = SpectralLaplacianHRPAllocator(
        tickers=TICKERS, fragility_penalty=0.25
    ).allocate(market_data, cov_matrix=cov_rd)
    pooler = EntropyPooler(n_scenarios=scenarios.n_paths)
    ent_res = pooler.fit()
    metrics = ScenarioEvaluator(ruin_threshold=-0.15).evaluate(
        scenarios,
        asset_weights=spec_res.weights.to_numpy(),
        entropy_result=ent_res,
    )
    print(
        f"Contagion Scenario Evaluation | Expected Return: {metrics.expected_return:.2%} | "
        f"CVaR(95%): {metrics.cvar_95:.2%}"
    )

    sim_html = output_dir / "1_reaction_diffusion_simulation_tearsheet.html"
    sim_ts = viz.SimulationTearSheet(
        scenarios,
        asset_weights=spec_res.weights,
        entropy_result=ent_res,
        ruin_threshold=-0.15,
        benchmark=bt_spectral,
    )
    sim_ts.to_html(sim_html, theme="alloq_dark")
    print(f"  [✓] Saved Simulation Tear Sheet to: {sim_html}")


if __name__ == "__main__":
    main()
