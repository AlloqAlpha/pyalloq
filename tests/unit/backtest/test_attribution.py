from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest import (
    AlmgrenChrissCostModel,
    AttributionResult,
    FlatBpsCostModel,
    PerformanceAttributor,
    RollingWindowSplitter,
    WalkForwardEngine,
)
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline

from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.dl.costs import TorchCostAdapter
from pyalloq.ml.optimizers.hrp import HRPAllocator


class TestBrinsonAndFactorAttribution:
    def test_brinson_fachler_exact_identity_and_3_tier_clusters(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        pipe = StrategyPipeline(
            allocator=HRPAllocator(tickers=sample_assets),
            cov_estimator=LedoitWolfShrinkage(),
        )
        engine = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=FlatBpsCostModel(bps=5.0),
            rebalance_freq="ME",
        )
        bt_res = engine.run(synthetic_market_data)

        attributor = PerformanceAttributor(linking_method="carino", n_auto_clusters=2)

        # Tier 1: Explicit sector_map
        explicit_map = {
            sample_assets[0]: "Tech",
            sample_assets[1]: "Tech",
            sample_assets[2]: "Defensive",
            sample_assets[3]: "Defensive",
        }
        brin_carino = attributor.compute_brinson_fachler(
            bt_res,
            synthetic_market_data,
            sector_map=explicit_map,
            linking_method="carino",
        )
        # Verify exact additive identity across effects: Allocation + Selection + Interaction == Total Active Return
        sum_effects_carino = (
            brin_carino.total_allocation_effect
            + brin_carino.total_selection_effect
            + brin_carino.total_interaction_effect
        )
        assert np.isclose(
            sum_effects_carino, brin_carino.total_active_return, atol=1e-10
        )

        # Verify Arithmetic linking identity
        brin_arith = attributor.compute_brinson_fachler(
            bt_res,
            synthetic_market_data,
            sector_map=explicit_map,
            linking_method="arithmetic",
        )
        sum_effects_arith = (
            brin_arith.total_allocation_effect
            + brin_arith.total_selection_effect
            + brin_arith.total_interaction_effect
        )
        assert np.isclose(sum_effects_arith, brin_arith.total_active_return, atol=1e-10)

        # Tier 2: Cross-sectional metadata
        cs_df = pd.DataFrame(
            {"sector": ["Growth", "Growth", "Value", "Macro"]},
            index=sample_assets,
        )
        md_cs = MarketData(
            prices=synthetic_market_data.prices,
            features=synthetic_market_data.features,
            cross_sectional=cs_df,
        )
        map_t2 = attributor.resolve_sector_map(md_cs, assets=sample_assets)
        assert map_t2[sample_assets[0]] == "Growth"
        assert map_t2[sample_assets[3]] == "Macro"

        # Tier 3: Automatic hierarchical correlation clustering when no metadata exists
        md_bare = MarketData(prices=synthetic_market_data.prices)
        map_t3 = attributor.resolve_sector_map(md_bare, assets=sample_assets)
        assert set(map_t3.keys()) == set(sample_assets)
        assert all(v.startswith("Cluster_") for v in map_t3.values())

    def test_factor_implied_attribution_recovers_synthetic_exposures(
        self, synthetic_market_data: MarketData
    ) -> None:
        dates = synthetic_market_data.prices.pct_change().dropna().index
        rng = np.random.default_rng(123)
        f_mkt = pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)
        f_mom = pd.Series(rng.normal(0.0002, 0.008, len(dates)), index=dates)
        factors_df = pd.DataFrame({"Market": f_mkt, "Momentum": f_mom}, index=dates)

        # Construct synthetic active return with true Alpha=0.0002/day, Beta_Mkt=1.25, Beta_Mom=-0.50
        noise = pd.Series(rng.normal(0.0, 0.0005, len(dates)), index=dates)
        active_ret = 0.0002 + 1.25 * f_mkt - 0.50 * f_mom + noise

        attributor = PerformanceAttributor(hac_lags=3)
        f_res = attributor.compute_factor_attribution(
            active_ret,
            synthetic_market_data,
            benchmark_returns=pd.Series(0.0, index=dates),
            factor_returns=factors_df,
        )

        assert f_res.r_squared > 0.90
        est_mkt = float(f_res.factor_summary.loc["Market", "exposure"])
        est_mom = float(f_res.factor_summary.loc["Momentum", "exposure"])
        assert np.isclose(est_mkt, 1.25, atol=0.08)
        assert np.isclose(est_mom, -0.50, atol=0.08)
        assert np.isclose(f_res.annualized_alpha, 0.0002 * 252.0, atol=0.03)

    def test_frictional_drag_accounting_and_almgren_chriss_decomposition(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        ac_cost = AlmgrenChrissCostModel(
            portfolio_aum=5_000_000.0, spread_bps=4.0, gamma=0.12
        )
        pipe = StrategyPipeline(
            allocator=MarkowitzAllocator(tickers=sample_assets),
            returns_estimator=EWMAReturnEstimator(span=15),
            cov_estimator=LedoitWolfShrinkage(),
        )
        engine = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=ac_cost,
            rebalance_freq="ME",
            rebalance_threshold=0.05,
        )
        bt_res = engine.run(synthetic_market_data)

        assert bt_res.gross_returns is not None
        assert bt_res.paper_returns is not None
        assert bt_res.cost_breakdown is not None

        attributor = PerformanceAttributor()
        full_attr = attributor.attribute(bt_res, synthetic_market_data)
        assert isinstance(full_attr, AttributionResult)

        wf = full_attr.frictional.waterfall_summary
        # Verify exact Gross-to-Net Waterfall bridge:
        # paper + drift_decay + spread_drag + impact_drag + borrow_drag == net_realized
        reconstructed_net = (
            float(wf["theoretical_paper_return"])
            + float(wf["alpha_decay_and_drift_drag"])
            + float(wf["spread_and_commission_drag"])
            + float(wf["market_impact_drag"])
            + float(wf["short_borrow_drag"])
        )
        assert np.isclose(
            reconstructed_net, float(wf["net_realized_return"]), atol=1e-10
        )
        assert float(wf["spread_and_commission_drag"]) < 0.0
        assert float(wf["market_impact_drag"]) < 0.0

        summary_df = full_attr.summary()
        assert bt_res.name in summary_df.columns
        assert "market_impact_drag" in summary_df.index

        # Verify differentiable PyTorch tensor decomposition via TorchCostAdapter
        adapter = TorchCostAdapter(ac_cost)
        w_delta = torch.tensor([0.10, -0.10, 0.05, -0.05], requires_grad=True)
        vol = torch.tensor([1e6, 2e6, 1.5e6, 3e6])
        spread_t, impact_t = adapter.decompose(w_delta, volume=vol)
        total_t = adapter(w_delta, volume=vol)
        assert torch.allclose(spread_t + impact_t, total_t, atol=1e-7)
        loss = spread_t.sum() + impact_t.sum()
        loss.backward()
        assert w_delta.grad is not None
