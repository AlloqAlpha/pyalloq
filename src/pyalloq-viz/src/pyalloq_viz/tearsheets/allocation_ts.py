from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult, OptimizationResult

from pyalloq_viz.allocation import (
    plot_correlation_and_periodic_returns,
    plot_hierarchical_composition,
    plot_multi_universe_frontier,
    plot_risk_contributions,
    plot_weight_drift_turnover,
)
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme
from pyalloq_viz.utils import compute_risk_contributions, reconstruct_drifted_weights


class AllocationTearSheet(BaseTearSheet):
    """
    4. Allocation Mechanics & Risk Tear Sheet:
       - Multi-Universe Efficient Frontier & Optimal Cloud
       - Weight Drift & Rebalance Turnover Dynamics
       - Hierarchical Composition Treemap & Target vs. Current Allocation
       - Risk Contribution Parity vs. Capital Weights
       - Asset Correlation Matrix & Monthly Periodic Return Heatmap
    """

    title = "Allocation Mechanics & Risk Tear Sheet"
    subtitle = "Efficient Frontier, Weight Drift & Turnover, Hierarchical Composition, Risk Parity & Seasonal Heatmaps"

    def __init__(
        self,
        optimization_result: OptimizationResult | pd.Series | None = None,
        backtest_result: BacktestResult | None = None,
        market_data: MarketData | None = None,
        cov_matrix: pd.DataFrame | np.ndarray | None = None,
        expected_returns: pd.Series | np.ndarray | None = None,
        scenario_data: ScenarioMarketData | None = None,
        allocator_or_pipeline: BaseAllocator | StrategyPipeline | None = None,
        target_weights: OptimizationResult | pd.Series | None = None,
        hierarchy: Mapping[str, str | Sequence[str]] | pd.DataFrame | None = None,
    ) -> None:
        self.optimization_result = optimization_result
        self.backtest_result = backtest_result
        self.market_data = market_data
        self.cov_matrix = cov_matrix
        self.expected_returns = expected_returns
        self.scenario_data = scenario_data
        self.allocator_or_pipeline = allocator_or_pipeline
        self.target_weights = target_weights
        self.hierarchy = hierarchy

        # Infer missing pieces automatically from market_data or backtest_result
        if self.cov_matrix is None and self.market_data is not None:
            rets = self.market_data.get_returns()
            self.cov_matrix = rets.cov() * 252.0
            if self.expected_returns is None:
                self.expected_returns = rets.mean() * 252.0

        if self.optimization_result is None and self.backtest_result is not None:
            last_w = self.backtest_result.weights.dropna(how="all").iloc[-1]
            self.optimization_result = OptimizationResult(
                weights=last_w, status=self.backtest_result.name
            )

    def _active_weights_series(self) -> pd.Series:
        if isinstance(self.optimization_result, OptimizationResult):
            return self.optimization_result.weights
        if isinstance(self.optimization_result, pd.Series):
            return self.optimization_result
        if self.backtest_result is not None and not self.backtest_result.weights.empty:
            return self.backtest_result.weights.dropna(how="all").iloc[-1]
        if self.cov_matrix is not None and isinstance(self.cov_matrix, pd.DataFrame):
            n = len(self.cov_matrix.columns)
            return pd.Series(1.0 / n, index=self.cov_matrix.columns)
        return pd.Series(dtype=float)

    def get_kpis(self) -> dict[str, str]:
        kpis: dict[str, str] = {}
        w_ser = self._active_weights_series()
        if len(w_ser) > 0:
            w_arr = w_ser.to_numpy(dtype=np.float64)
            eff_n = float(1.0 / max(float(np.sum(np.square(w_arr))), 1e-12))
            kpis["Active Assets"] = str(int((w_arr > 1e-4).sum()))
            kpis["Effective Bets (1/∑w²)"] = f"{eff_n:.2f}"
            kpis["Max Asset Weight"] = f"{float(np.max(w_arr)):.1%}"

        if self.cov_matrix is not None and len(w_ser) > 0:
            rc = compute_risk_contributions(w_ser, self.cov_matrix)
            pcrc = np.clip(rc["pcrc"].to_numpy(dtype=np.float64), 1e-12, None)
            encb = float(1.0 / np.sum(np.square(pcrc)))
            kpis["Effective Risk Bets (ENCB)"] = f"{encb:.2f}"
            kpis["Ex-Ante Volatility"] = f"{float(np.sum(rc['crc'])):.2%}"

        if self.backtest_result is not None:
            _, turnover = reconstruct_drifted_weights(
                self.backtest_result, market_data=self.market_data
            )
            if len(turnover) > 1:
                kpis["Cumulative Turnover"] = f"{float(turnover.iloc[1:].sum()):.1%}"
            if "sharpe_ratio" in self.backtest_result.tear_sheet.index:
                val = self.backtest_result.tear_sheet.loc["sharpe_ratio", "Value"]
                sr = float(val)  # type: ignore[arg-type]
                kpis["Realized Sharpe"] = f"{sr:.2f}"

        return kpis

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        figs: list[go.Figure] = []
        w_ser = self._active_weights_series()

        if self.cov_matrix is not None or self.scenario_data is not None:
            figs.append(
                plot_multi_universe_frontier(
                    expected_returns=self.expected_returns,
                    cov_matrix=self.cov_matrix,
                    scenario_data=self.scenario_data,
                    allocator_or_pipeline=self.allocator_or_pipeline,
                    optimal_portfolios=self.optimization_result
                    if isinstance(self.optimization_result, OptimizationResult)
                    else None,
                    theme=theme,
                )
            )

        if len(w_ser) > 0 and self.cov_matrix is not None:
            figs.append(
                plot_risk_contributions(
                    self.optimization_result
                    if self.optimization_result is not None
                    else w_ser,
                    cov_matrix=self.cov_matrix,
                    theme=theme,
                )
            )

        if len(w_ser) > 0:
            figs.append(
                plot_hierarchical_composition(
                    self.optimization_result
                    if self.optimization_result is not None
                    else w_ser,
                    target_or_benchmark_weights=self.target_weights,
                    hierarchy=self.hierarchy,
                    market_data=self.market_data,
                    theme=theme,
                )
            )

        if self.backtest_result is not None:
            figs.append(
                plot_weight_drift_turnover(
                    self.backtest_result,
                    market_data=self.market_data,
                    theme=theme,
                )
            )

        if self.market_data is not None or self.cov_matrix is not None:
            md_or_cov = (
                self.market_data if self.market_data is not None else self.cov_matrix
            )
            if md_or_cov is not None:
                figs.append(
                    plot_correlation_and_periodic_returns(
                        md_or_cov,
                        portfolio_returns=self.backtest_result,
                        theme=theme,
                    )
                )

        return figs

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        w_ser = self._active_weights_series()

        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Multi-Universe Efficient Frontier",
                "2. Capital Weights vs. Component Risk Contributions",
                "3. Weight Drift Dynamics"
                if self.backtest_result is not None
                else "3. Target vs. Current Allocation",
                "4. Asset Correlation Matrix Heatmap",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Panel (1, 1): Efficient Frontier
        if self.cov_matrix is not None or self.scenario_data is not None:
            fig_ef = plot_multi_universe_frontier(
                expected_returns=self.expected_returns,
                cov_matrix=self.cov_matrix,
                scenario_data=self.scenario_data,
                allocator_or_pipeline=self.allocator_or_pipeline,
                optimal_portfolios=self.optimization_result
                if isinstance(self.optimization_result, OptimizationResult)
                else None,
                theme=theme,
            )
            for trace in fig_ef.data:
                if hasattr(trace, "marker") and getattr(
                    trace.marker, "showscale", None
                ):
                    trace.marker.showscale = False
                combined.add_trace(trace, row=1, col=1)

        # Panel (1, 2): Risk Contributions
        if len(w_ser) > 0 and self.cov_matrix is not None:
            fig_rc = plot_risk_contributions(
                self.optimization_result
                if self.optimization_result is not None
                else w_ser,
                cov_matrix=self.cov_matrix,
                theme=theme,
            )
            for trace in fig_rc.data:
                combined.add_trace(trace, row=1, col=2)

        # Panel (2, 1): Weight Drift or Allocation Bars
        if self.backtest_result is not None:
            fig_wd = plot_weight_drift_turnover(
                self.backtest_result, market_data=self.market_data, theme=theme
            )
            for trace in fig_wd.data:
                if getattr(trace, "yaxis", "y") == "y":
                    combined.add_trace(trace, row=2, col=1)
        elif len(w_ser) > 0:
            fig_hc = plot_hierarchical_composition(
                self.optimization_result
                if self.optimization_result is not None
                else w_ser,
                target_or_benchmark_weights=self.target_weights,
                hierarchy=self.hierarchy,
                market_data=self.market_data,
                theme=theme,
            )
            for trace in fig_hc.data:
                if isinstance(trace, go.Bar):
                    combined.add_trace(trace, row=2, col=1)

        # Panel (2, 2): Correlation Matrix Heatmap
        if self.market_data is not None or self.cov_matrix is not None:
            md_or_cov = (
                self.market_data if self.market_data is not None else self.cov_matrix
            )
            if md_or_cov is not None:
                fig_corr = plot_correlation_and_periodic_returns(
                    md_or_cov,
                    portfolio_returns=self.backtest_result,
                    theme=theme,
                )
                if len(fig_corr.data) > 0:
                    combined.add_trace(fig_corr.data[0], row=2, col=2)

        kpis = self.get_kpis()
        kpi_summary = " | ".join(f"{k}: {v}" for k, v in list(kpis.items())[:5])
        apply_plotly_theme(
            combined,
            theme=theme,
            title=self.title,
            subtitle=kpi_summary,
            height=940,
        )
        return combined
