from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData
from pyalloq_core.results import BacktestResult, MultiPeriodOptimizationResult

from pyalloq_viz.allocation import plot_weight_drift_turnover
from pyalloq_viz.multi_period import (
    _compute_rolling_thermodynamics,
    plot_alpha_decay_surface,
    plot_mpc_trajectory,
    plot_mpo_pillar_comparison,
    plot_thermodynamic_regime_gate,
)
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme


class MultiPeriodTearSheet(BaseTearSheet):
    """
    5. Multi-Period Portfolio Optimization (MPO) Tear Sheet:
       - Planned H-Stage Weight Trajectory & Child-Order Trade Splitting
       - Multi-Horizon Alpha Decay Term Structure (Gârleanu-Pedersen)
       - Thermodynamic & Bures-Wasserstein Regime Gate Diagnostics (Moroke et al., 2026)
       - Cross-Pillar Net-of-Cost Equity, Drawdown & Cumulative Turnover Comparison
    """

    title = "Multi-Period Portfolio Optimization (MPO) Tear Sheet"
    subtitle = (
        "Receding-Horizon Trajectory, Alpha Decay Term Structure, "
        "Thermodynamic Regime Gate & 3-Pillar Net-of-Cost Attribution"
    )

    def __init__(
        self,
        mpo_result: MultiPeriodOptimizationResult | None = None,
        horizon_forecasts: pd.DataFrame | None = None,
        market_data: MarketData | None = None,
        strategy_comparison: Mapping[str, BacktestResult] | None = None,
        backtest_result: BacktestResult | None = None,
        current_weights: pd.Series | None = None,
        fast_window: int = 15,
        slow_window: int = 45,
        w2_threshold: float = 0.025,
    ) -> None:
        self.mpo_result = mpo_result
        self.horizon_forecasts = horizon_forecasts
        self.market_data = market_data
        self.strategy_comparison = strategy_comparison
        self.backtest_result = backtest_result
        self.current_weights = current_weights
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.w2_threshold = w2_threshold

    def get_kpis(self) -> dict[str, str]:
        kpis: dict[str, str] = {}

        if self.mpo_result is not None:
            kpis["MPO Solver Status"] = str(self.mpo_result.status)
            if self.mpo_result.weights_trajectory is not None:
                kpis["Planning Horizon"] = (
                    f"{len(self.mpo_result.weights_trajectory)} stages"
                )
            if self.mpo_result.trades_trajectory is not None:
                z_np = self.mpo_result.trades_trajectory.to_numpy(dtype=np.float64)
                kpis["Immediate Step Turnover"] = (
                    f"{float(np.sum(np.abs(z_np[0]))):.2%}"
                )
                kpis["Planned Horizon Turnover"] = f"{float(np.sum(np.abs(z_np))):.2%}"

        if self.market_data is not None:
            diag_df = _compute_rolling_thermodynamics(
                self.market_data,
                fast_window=self.fast_window,
                slow_window=self.slow_window,
                w2_threshold=self.w2_threshold,
            )
            if len(diag_df) > 0:
                last_row = diag_df.iloc[-1]
                kpis["Current Bures-W₂"] = (
                    f"{float(last_row['wasserstein_distance']):.4f}"
                )
                kpis["Carnot Efficiency (η)"] = (
                    f"{float(last_row['carnot_efficiency']):.1%}"
                )
                kpis["Regime Gate Scale (λ)"] = (
                    f"{float(last_row['damping_factor']):.1%}"
                )

        primary_bt = self.backtest_result
        if primary_bt is None and self.strategy_comparison:
            primary_bt = next(iter(self.strategy_comparison.values()))

        if primary_bt is not None and "sharpe_ratio" in primary_bt.tear_sheet.index:
            sr = float(primary_bt.tear_sheet.loc["sharpe_ratio", "Value"])  # type: ignore[arg-type]
            kpis["Realized Net Sharpe"] = f"{sr:.2f}"

        return kpis

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        figs: list[go.Figure] = []

        if self.mpo_result is not None:
            figs.append(
                plot_mpc_trajectory(
                    self.mpo_result,
                    current_weights=self.current_weights,
                    theme=theme,
                )
            )

        if self.horizon_forecasts is not None:
            figs.append(
                plot_alpha_decay_surface(
                    self.horizon_forecasts,
                    theme=theme,
                )
            )

        if self.market_data is not None:
            figs.append(
                plot_thermodynamic_regime_gate(
                    self.market_data,
                    fast_window=self.fast_window,
                    slow_window=self.slow_window,
                    w2_threshold=self.w2_threshold,
                    theme=theme,
                )
            )

        if self.strategy_comparison:
            figs.append(
                plot_mpo_pillar_comparison(
                    self.strategy_comparison,
                    theme=theme,
                )
            )
        elif self.backtest_result is not None:
            figs.append(
                plot_weight_drift_turnover(
                    self.backtest_result,
                    market_data=self.market_data,
                    theme=theme,
                )
            )

        return figs

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Planned Multi-Stage Weight Trajectory",
                "2. Multi-Horizon Alpha Decay Trajectories",
                "3. Bures-Wasserstein Regime Distance W₂(t)",
                "4. Multi-Pillar Net-of-Cost Equity Comparison",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Panel (1, 1): MPC Trajectory weights
        if self.mpo_result is not None:
            fig_traj = plot_mpc_trajectory(
                self.mpo_result,
                current_weights=self.current_weights,
                theme=theme,
            )
            for trace in fig_traj.data:
                if getattr(trace, "yaxis", "y") == "y":
                    combined.add_trace(trace, row=1, col=1)

        # Panel (1, 2): Alpha Decay trajectories
        if self.horizon_forecasts is not None:
            fig_alpha = plot_alpha_decay_surface(self.horizon_forecasts, theme=theme)
            for trace in fig_alpha.data:
                if isinstance(trace, go.Scatter):
                    combined.add_trace(trace, row=1, col=2)

        # Panel (2, 1): Bures-Wasserstein W2 distance
        if self.market_data is not None:
            fig_thermo = plot_thermodynamic_regime_gate(
                self.market_data,
                fast_window=self.fast_window,
                slow_window=self.slow_window,
                w2_threshold=self.w2_threshold,
                theme=theme,
            )
            for trace in fig_thermo.data:
                if getattr(trace, "yaxis", "y") == "y":
                    combined.add_trace(trace, row=2, col=1)

        # Panel (2, 2): Pillar Equity Comparison
        if self.strategy_comparison:
            fig_comp = plot_mpo_pillar_comparison(self.strategy_comparison, theme=theme)
            for trace in fig_comp.data:
                if getattr(trace, "yaxis", "y") == "y":
                    combined.add_trace(trace, row=2, col=2)

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
