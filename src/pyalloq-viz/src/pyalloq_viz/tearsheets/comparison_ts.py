from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_backtest.attribution import AttributionResult, PerformanceAttributor
from pyalloq_backtest.benchmark import (
    BacktestTracker,
    ComparisonResult,
    MultiStrategyComparator,
)
from pyalloq_backtest.metrics import compute_deflated_sharpe_ratio
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult

from pyalloq_viz.comparison import (
    plot_attribution_waterfall,
    plot_equity_drawdown_overlay,
    plot_multi_strategy_fan_chart,
    plot_multi_turnover_drift,
)
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme


class ComparisonTearSheet(BaseTearSheet):
    """
    Multi-Strategy Trajectory & Performance Attribution Tear Sheet:
      1. Cumulative Net-of-Cost Equity & Underwater Drawdown Overlay
      2. Multi-Strategy Probabilistic Cone / Fan Chart & Terminal Wealth KDE
      3. Intra-Rebalance Sawtooth Weight Drift & Two-Way Turnover Dynamics
      4. Brinson-Fachler Active Decomposition, Factor Alpha & Frictional Drag Waterfall
      5. Summary KPIs including Deflated Sharpe Ratio (DSR) & Multiple Testing Count (N)
    """

    title = "Multi-Strategy Comparison & Attribution Tear Sheet"
    subtitle = "Deflated Sharpe Ratio (DSR), Trajectory Cones, Sawtooth Drift & Brinson-Fachler / Factor Attribution"

    def __init__(
        self,
        comparison_or_results: ComparisonResult | Mapping[str, BacktestResult],
        market_data: MarketData | None = None,
        scenario_data: ScenarioMarketData | None = None,
        benchmark: BacktestResult | pd.Series | None = None,
        attributions: Mapping[str, AttributionResult] | None = None,
    ) -> None:
        self.market_data = market_data
        self.scenario_data = scenario_data
        self.benchmark = benchmark

        if isinstance(comparison_or_results, ComparisonResult):
            self.comparison_result: ComparisonResult | None = comparison_or_results
            self.results: dict[str, BacktestResult] = dict(
                comparison_or_results.results
            )
            self.attributions: dict[str, AttributionResult] = (
                dict(attributions)
                if attributions is not None
                else dict(comparison_or_results.attributions)
            )
            self.scenario_results: dict[str, MonteCarloResult] | None = (
                comparison_or_results.scenario_results
            )
            self.n_trials = comparison_or_results.n_trials
        else:
            self.comparison_result = None
            self.results = dict(comparison_or_results)
            self.attributions = dict(attributions) if attributions is not None else {}
            if not self.attributions and self.market_data is not None:
                attributor = PerformanceAttributor()
                for name, bt_res in self.results.items():
                    self.attributions[name] = attributor.attribute(
                        bt_res,
                        self.market_data,
                        benchmark_returns=self.benchmark,
                    )
            self.scenario_results = None
            if self.scenario_data is not None:
                self.scenario_results = {
                    name: MultiStrategyComparator._evaluate_on_scenarios(
                        bt_res, self.scenario_data
                    )
                    for name, bt_res in self.results.items()
                }
            self.n_trials = max(
                BacktestTracker.instance().n_trials, len(self.results), 1
            )

    def get_kpis(self) -> dict[str, str]:
        kpis: dict[str, str] = {
            "Strategies Evaluated": str(len(self.results)),
            "Trials Accounted (N)": str(self.n_trials),
        }
        if not self.results:
            return kpis

        sharpes: dict[str, float] = {}
        dsrs: dict[str, float] = {}
        mdds: dict[str, float] = {}

        all_srs: list[float] = []
        for name, bt_res in self.results.items():
            if "sharpe_ratio" in bt_res.tear_sheet.index:
                sr = float(bt_res.tear_sheet.loc["sharpe_ratio", "Value"])  # type: ignore[arg-type]
            else:
                r = bt_res.returns.dropna()
                std_v = float(r.std() * np.sqrt(252.0)) if len(r) > 1 else 0.0
                sr = float(r.mean() * 252.0 / std_v) if std_v > 1e-12 else 0.0
            sharpes[name] = sr
            all_srs.append(sr)

            if "maximum_drawdown" in bt_res.tear_sheet.index:
                mdds[name] = float(bt_res.tear_sheet.loc["maximum_drawdown", "Value"])  # type: ignore[arg-type]
            else:
                eq = bt_res.equity_curve.dropna()
                rmax = eq.cummax().replace(0.0, np.nan)
                mdds[name] = float(((eq - rmax) / rmax).min()) if len(eq) > 0 else 0.0

        for name, bt_res in self.results.items():
            if (
                "deflated_sharpe_ratio" in bt_res.tear_sheet.index
                and len(self.results) == 1
            ):
                dsrs[name] = float(
                    bt_res.tear_sheet.loc["deflated_sharpe_ratio", "Value"]  # type: ignore[arg-type]
                )
            else:
                dsr_v, _ = compute_deflated_sharpe_ratio(
                    bt_res.returns,
                    n_trials=self.n_trials,
                    trial_sharpes=all_srs,
                )
                dsrs[name] = dsr_v

        best_sr_name = max(sharpes, key=lambda k: sharpes[k])
        best_dsr_name = max(dsrs, key=lambda k: dsrs[k])
        best_mdd_name = max(mdds, key=lambda k: mdds[k])

        kpis["Top Net Sharpe"] = f"{best_sr_name} ({sharpes[best_sr_name]:.2f})"
        kpis["Top Deflated Sharpe (DSR)"] = (
            f"{best_dsr_name} ({dsrs[best_dsr_name]:.1%})"
        )
        kpis["Shallowest Drawdown"] = f"{best_mdd_name} ({mdds[best_mdd_name]:.1%})"

        if self.attributions:
            best_alpha_name = max(
                self.attributions,
                key=lambda k: self.attributions[k].factor.annualized_alpha,
            )
            alpha_v = self.attributions[best_alpha_name].factor.annualized_alpha
            kpis["Top Factor Alpha"] = f"{best_alpha_name} ({alpha_v:+.2%})"

        return kpis

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        source_obj: ComparisonResult | dict[str, BacktestResult] = (
            self.comparison_result
            if self.comparison_result is not None
            else self.results
        )
        figs: list[go.Figure] = [
            plot_equity_drawdown_overlay(
                source_obj,
                benchmark=self.benchmark,
                theme=theme,
            ),
            plot_multi_strategy_fan_chart(
                source_obj,
                scenario_data=self.scenario_data,
                theme=theme,
            ),
            plot_multi_turnover_drift(
                source_obj,
                market_data=self.market_data,
                theme=theme,
            ),
        ]
        if self.attributions or self.market_data is not None:
            attr_input = (
                self.comparison_result
                if (
                    self.comparison_result is not None
                    and self.comparison_result.attributions
                )
                else (self.attributions if self.attributions else self.results)
            )
            figs.append(
                plot_attribution_waterfall(
                    attr_input,
                    market_data=self.market_data,
                    theme=theme,
                )
            )
        return figs

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        source_obj: ComparisonResult | dict[str, BacktestResult] = (
            self.comparison_result
            if self.comparison_result is not None
            else self.results
        )

        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Cumulative Net-of-Cost Equity Overlay",
                "2. Multi-Strategy Probabilistic Equity Cones",
                "3. Intra-Rebalance Sawtooth Weight Drift",
                "4. Brinson-Fachler & Factor Attribution",
            ),
            vertical_spacing=0.13,
            horizontal_spacing=0.08,
        )

        # Panel (1, 1): Equity Curves from plot_equity_drawdown_overlay (top row traces)
        fig_eq = plot_equity_drawdown_overlay(
            source_obj,
            benchmark=self.benchmark,
            highlight_shock_window=False,
            theme=theme,
        )
        for trace in fig_eq.data:
            if getattr(trace, "yaxis", "y") == "y":
                combined.add_trace(trace, row=1, col=1)

        # Panel (1, 2): Fan Chart Cones from plot_multi_strategy_fan_chart (left col traces)
        fig_fan = plot_multi_strategy_fan_chart(
            source_obj, scenario_data=self.scenario_data, theme=theme
        )
        for trace in fig_fan.data:
            if getattr(trace, "xaxis", "x") == "x":
                trace.showlegend = False
                combined.add_trace(trace, row=1, col=2)

        # Panel (2, 1): Sawtooth Weight Drift from plot_multi_turnover_drift (top row traces)
        fig_drift = plot_multi_turnover_drift(
            source_obj, market_data=self.market_data, theme=theme
        )
        for trace in fig_drift.data:
            if getattr(trace, "yaxis", "y") == "y":
                trace.showlegend = False
                combined.add_trace(trace, row=2, col=1)

        # Panel (2, 2): Brinson-Fachler & Factor Attribution Bars
        if self.attributions or self.market_data is not None:
            attr_input = (
                self.comparison_result
                if (
                    self.comparison_result is not None
                    and self.comparison_result.attributions
                )
                else (self.attributions if self.attributions else self.results)
            )
            fig_attr = plot_attribution_waterfall(
                attr_input, market_data=self.market_data, theme=theme
            )
            for trace in fig_attr.data:
                if getattr(trace, "xaxis", "x") == "x":
                    combined.add_trace(trace, row=2, col=2)

        kpis = self.get_kpis()
        kpi_summary = " | ".join(f"{k}: {v}" for k, v in list(kpis.items())[:5])
        apply_plotly_theme(
            combined,
            theme=theme,
            title=self.title,
            subtitle=kpi_summary,
            height=920,
        )
        combined.update_layout(barmode="group")
        combined.update_yaxes(title_text="Equity (x)", row=1, col=1)
        combined.update_yaxes(title_text="Cone Equity (x)", row=1, col=2)
        combined.update_yaxes(title_text="L1 Drift (%)", tickformat=".1%", row=2, col=1)
        combined.update_yaxes(
            title_text="Active Effect (%)", tickformat=".1%", row=2, col=2
        )
        return combined
