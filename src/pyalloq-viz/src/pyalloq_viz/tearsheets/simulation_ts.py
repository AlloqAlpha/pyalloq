from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_core.data import ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult
from pyalloq_simulation.views.pooler import EntropyResult

from pyalloq_viz.simulation import (
    plot_cvar_tail_risk,
    plot_drawdown_fan_chart,
    plot_fan_chart,
    plot_terminal_wealth_ruin,
)
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme
from pyalloq_viz.utils import (
    compute_drawdown_paths,
    extract_equity_paths,
    weighted_quantiles,
)


class SimulationTearSheet(BaseTearSheet):
    """
    1. The Multiverse & Simulation Tear Sheet:
       - Probabilistic Cone (Fan) Chart
       - Terminal Wealth & Ruin Histogram
       - CVaR Tail-Risk Shading
       - Underwater Drawdown Fan Chart
    """

    title = "Multiverse & Simulation Tear Sheet"
    subtitle = "Synthetic Path Dispersion, Terminal Ruin Mass, CVaR Tail Shortfall & Underwater Drawdown Cones"

    def __init__(
        self,
        source: MonteCarloResult | ScenarioMarketData | pd.DataFrame | np.ndarray,
        asset_weights: np.ndarray | pd.Series | None = None,
        entropy_result: EntropyResult | None = None,
        ruin_threshold: float = -0.20,
        cvar_alpha: float = 0.05,
        benchmark: BacktestResult | pd.Series | None = None,
    ) -> None:
        self.source = source
        self.asset_weights = asset_weights
        self.entropy_result = entropy_result
        self.ruin_threshold = ruin_threshold
        self.cvar_alpha = cvar_alpha
        self.benchmark = benchmark

    def get_kpis(self) -> dict[str, str]:
        eq_paths, _ = extract_equity_paths(
            self.source, asset_weights=self.asset_weights
        )
        M, T = eq_paths.shape
        probs = (
            np.asarray(self.entropy_result.weights, dtype=np.float64)
            if self.entropy_result is not None
            else np.full(M, 1.0 / max(1, M))
        )
        probs = probs / np.sum(probs)

        term_rets = eq_paths[:, -1] - 1.0
        exp_ret = float(np.dot(term_rets, probs))
        vol = float(np.sqrt(max(float(np.dot(probs, (term_rets - exp_ret) ** 2)), 0.0)))
        sharpe = exp_ret / vol if vol > 1e-12 else 0.0

        var_val = float(
            weighted_quantiles(term_rets, [self.cvar_alpha], sample_weight=probs)[0]
        )
        tail_mask = term_rets <= var_val
        if np.any(tail_mask) and np.sum(probs[tail_mask]) > 0:
            cvar_val = float(
                np.dot(
                    term_rets[tail_mask], probs[tail_mask] / np.sum(probs[tail_mask])
                )
            )
        else:
            cvar_val = float(np.min(term_rets))

        dd_paths = compute_drawdown_paths(eq_paths)
        path_mdds = np.min(dd_paths, axis=1)
        exp_mdd = float(np.dot(probs, path_mdds))

        ruin_drop = -abs(self.ruin_threshold)
        ruin_mask = (path_mdds <= ruin_drop) | ((eq_paths[:, -1] - 1.0) <= ruin_drop)
        prob_ruin = float(np.dot(probs, ruin_mask.astype(np.float64)))

        return {
            "Synthetic Paths (M)": f"{M:,}",
            "Horizon Steps (T)": f"{T:,}",
            "Expected Horizon Return": f"{exp_ret:.2%}",
            "Horizon Sharpe": f"{sharpe:.2f}",
            f"CVaR ({int((1 - self.cvar_alpha) * 100)}%)": f"{cvar_val:.2%}",
            "Expected Max Drawdown": f"{exp_mdd:.2%}",
            f"Prob. of Ruin ({ruin_drop:.0%})": f"{prob_ruin:.2%}",
        }

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        return [
            plot_fan_chart(
                self.source,
                asset_weights=self.asset_weights,
                entropy_result=self.entropy_result,
                benchmark=self.benchmark,
                theme=theme,
            ),
            plot_terminal_wealth_ruin(
                self.source,
                asset_weights=self.asset_weights,
                entropy_result=self.entropy_result,
                ruin_threshold=self.ruin_threshold,
                theme=theme,
            ),
            plot_cvar_tail_risk(
                self.source,
                asset_weights=self.asset_weights,
                entropy_result=self.entropy_result,
                alpha=self.cvar_alpha,
                theme=theme,
            ),
            plot_drawdown_fan_chart(
                self.source,
                asset_weights=self.asset_weights,
                entropy_result=self.entropy_result,
                theme=theme,
            ),
        ]

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        figs = self.get_component_figures(theme=theme)
        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Probabilistic Equity Cone (Fan Chart)",
                "2. Terminal Wealth & Ruin Distribution",
                "3. Return Distribution & CVaR Tail-Risk",
                "4. Underwater Drawdown Fan Chart",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for sub_fig, (r, c) in zip(figs, positions):
            for trace in sub_fig.data:
                combined.add_trace(trace, row=r, col=c)

        kpis = self.get_kpis()
        kpi_summary = " | ".join(f"{k}: {v}" for k, v in list(kpis.items())[:5])
        apply_plotly_theme(
            combined,
            theme=theme,
            title=self.title,
            subtitle=kpi_summary,
            height=920,
        )
        combined.update_yaxes(tickformat=".1%", row=2, col=2)
        combined.update_xaxes(tickformat=".1%", row=2, col=1)
        return combined
