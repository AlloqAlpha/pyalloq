from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_backtest.overfitting import (
    CPCVResult,
    CPCVSplitter,
    PBOEvaluator,
    PBOResult,
)
from pyalloq_core.results import BacktestResult, MonteCarloResult

from pyalloq_viz.robustness import plot_cpcv_paths, plot_pbo_distribution
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme


class RobustnessTearSheet(BaseTearSheet):
    """
    2. Robustness & Overfitting Tear Sheet:
       - Probability of Backtest Overfitting (PBO) & IS/OOS Sharpe Degradation
       - Combinatorial Scrambling (CPCV) Timeline Survival & Drawdown Distribution
    """

    title = "Robustness & Overfitting Tear Sheet"
    subtitle = "Probability of Backtest Overfitting (PBO) & Combinatorial Purged Scrambling (CPCV)"

    def __init__(
        self,
        backtest_or_pbo: BacktestResult
        | PBOResult
        | pd.DataFrame
        | np.ndarray
        | Sequence[BacktestResult],
        cpcv_or_synthetic: CPCVResult | MonteCarloResult | None = None,
        n_groups: int = 6,
        n_partitions: int = 8,
    ) -> None:
        self.n_groups = n_groups
        self.n_partitions = n_partitions

        # Resolve CPCVResult
        if isinstance(cpcv_or_synthetic, CPCVResult):
            self.cpcv_result = cpcv_or_synthetic
        elif isinstance(backtest_or_pbo, BacktestResult):
            self.cpcv_result = CPCVSplitter(n_groups=n_groups).scramble_timelines(
                backtest_or_pbo
            )
        elif isinstance(backtest_or_pbo, pd.DataFrame):
            # Use first column or mean strategy for CPCV scrambling
            self.cpcv_result = CPCVSplitter(n_groups=n_groups).scramble_timelines(
                backtest_or_pbo.iloc[:, 0]
            )
        elif (
            isinstance(backtest_or_pbo, Sequence)
            and len(backtest_or_pbo) > 0
            and isinstance(backtest_or_pbo[0], BacktestResult)
        ):
            self.cpcv_result = CPCVSplitter(n_groups=n_groups).scramble_timelines(
                backtest_or_pbo[0]
            )
        else:
            arr = np.asarray(backtest_or_pbo, dtype=np.float64)
            col0 = arr[:, 0] if arr.ndim == 2 else arr
            self.cpcv_result = CPCVSplitter(n_groups=n_groups).scramble_timelines(
                pd.Series(col0)
            )

        # Resolve PBOResult
        evaluator = PBOEvaluator(n_partitions=n_partitions)
        if isinstance(backtest_or_pbo, PBOResult):
            self.pbo_result = backtest_or_pbo
        elif isinstance(backtest_or_pbo, BacktestResult):
            comp = (
                cpcv_or_synthetic if cpcv_or_synthetic is not None else self.cpcv_result
            )
            self.pbo_result = evaluator.evaluate_benchmark(backtest_or_pbo, comp)
        else:
            self.pbo_result = evaluator.evaluate_cscv(backtest_or_pbo)

    def get_kpis(self) -> dict[str, str]:
        srs = self.cpcv_result.sharpe_ratios.to_numpy(dtype=np.float64)
        mdds = self.cpcv_result.max_drawdowns.to_numpy(dtype=np.float64)
        hist_sr = (
            f"{self.pbo_result.historical_sharpe:.2f}"
            if self.pbo_result.historical_sharpe is not None
            else "N/A"
        )
        return {
            "Evaluation Mode": self.pbo_result.mode.upper(),
            "PBO (Overfitting Prob)": f"{self.pbo_result.pbo:.1%}",
            "Prob. of OOS Loss": f"{self.pbo_result.prob_oos_loss:.1%}",
            "Historical / Best Sharpe": hist_sr,
            "CPCV Median Sharpe": f"{float(np.median(srs)):.2f}",
            "CPCV Survival (SR > 0)": f"{float(np.mean(srs > 0)):.1%}",
            "CPCV Median Max DD": f"{float(np.median(mdds)):.2%}",
        }

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        return [
            plot_pbo_distribution(self.pbo_result, theme=theme),
            plot_cpcv_paths(self.cpcv_result, theme=theme),
        ]

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        fig_pbo = plot_pbo_distribution(self.pbo_result, theme=theme)
        fig_cpcv = plot_cpcv_paths(self.cpcv_result, theme=theme)

        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. PBO Logit / Sharpe Distribution",
                "2. Out-of-Sample Degradation / ECDF",
                "3. Combinatorial Regime-Scrambled Timelines (CPCV)",
                "4. Scrambled Path Survival: Sharpe vs. Max Drawdown",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Extract traces by xaxis assignment from each 1x2 figure
        for trace in fig_pbo.data:
            col_idx = 2 if getattr(trace, "xaxis", "x") in ("x2", "x3") else 1
            combined.add_trace(trace, row=1, col=col_idx)

        for trace in fig_cpcv.data:
            col_idx = 2 if getattr(trace, "xaxis", "x") in ("x2", "x3") else 1
            combined.add_trace(trace, row=2, col=col_idx)

        kpis = self.get_kpis()
        kpi_summary = " | ".join(f"{k}: {v}" for k, v in list(kpis.items())[:5])
        apply_plotly_theme(
            combined,
            theme=theme,
            title=self.title,
            subtitle=kpi_summary,
            height=900,
        )
        return combined
