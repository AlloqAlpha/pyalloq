from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult
from pyalloq_simulation.generators.regimes import GMMRegimeGenerator
from pyalloq_simulation.views.pooler import EntropyResult

from pyalloq_viz.regime import plot_entropy_pooling, plot_regime_transitions
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import ThemeName, apply_plotly_theme


class MacroRegimeTearSheet(BaseTearSheet):
    """
    3. Macro Regime & Entropy Tear Sheet:
       - Prior vs. Posterior Density (Meucci's Entropy Pooling)
       - GMM Markov Regime Transition Overlays (Bull/Neutral/Bear State Probabilities)
    """

    title = "Macro Regime & Entropy Tear Sheet"
    subtitle = "Meucci Entropy Pooling Posterior Shifts & GMM Markov Regime Dynamics"

    def __init__(
        self,
        entropy_result: EntropyResult | None = None,
        equity_or_scenarios: BacktestResult
        | pd.Series
        | ScenarioMarketData
        | None = None,
        gmm_or_probs: GMMRegimeGenerator | pd.DataFrame | np.ndarray | None = None,
        scenario_data: ScenarioMarketData | None = None,
        market_data: MarketData | None = None,
        asset_weights: np.ndarray | pd.Series | None = None,
    ) -> None:
        self.entropy_result = entropy_result
        self.equity_or_scenarios = equity_or_scenarios or scenario_data
        self.gmm_or_probs = gmm_or_probs
        self.scenario_data = (
            scenario_data
            if scenario_data is not None
            else (
                equity_or_scenarios
                if isinstance(equity_or_scenarios, ScenarioMarketData)
                else None
            )
        )
        self.market_data = market_data
        self.asset_weights = asset_weights

    def get_kpis(self) -> dict[str, str]:
        kpis: dict[str, str] = {}
        if self.entropy_result is not None:
            post = np.asarray(self.entropy_result.weights, dtype=np.float64)
            M = len(post)
            safe_post = np.where(post > 1e-16, post, 1e-16)
            ens = float(np.exp(-np.sum(post * np.log(safe_post))))
            kpis["Total Scenarios (M)"] = f"{M:,}"
            kpis["KL Divergence"] = f"{self.entropy_result.kl_divergence:.4f}"
            kpis["Effective Scenarios (ENS)"] = f"{ens:.1f} ({ens / max(1, M):.1%})"
            kpis["Max Posterior Weight"] = f"{float(np.max(post)):.4f}"
            kpis["Entropy Solver Status"] = (
                "Converged" if self.entropy_result.success else "Fallback"
            )

        if (
            isinstance(self.gmm_or_probs, GMMRegimeGenerator)
            and self.gmm_or_probs.is_fitted
        ):
            kpis["GMM Regimes (K)"] = str(self.gmm_or_probs.k_eff)
            if (
                self.gmm_or_probs.transition_matrix is not None
                and self.gmm_or_probs.k_eff >= 1
            ):
                diag_stay = float(np.mean(np.diag(self.gmm_or_probs.transition_matrix)))
                kpis["Avg Regime Persistence"] = f"{diag_stay:.1%}"

        return kpis

    def get_component_figures(
        self, theme: ThemeName | str = "alloq_dark"
    ) -> list[go.Figure]:
        figs: list[go.Figure] = []
        if self.entropy_result is not None:
            figs.append(
                plot_entropy_pooling(
                    self.entropy_result,
                    scenario_data=self.scenario_data,
                    asset_weights=self.asset_weights,
                    theme=theme,
                )
            )
        if self.equity_or_scenarios is not None and (
            self.gmm_or_probs is not None
            or (
                isinstance(self.equity_or_scenarios, ScenarioMarketData)
                and "regime_states" in self.equity_or_scenarios.features
            )
        ):
            figs.append(
                plot_regime_transitions(
                    self.equity_or_scenarios,
                    gmm_or_probs=self.gmm_or_probs,
                    market_data=self.market_data,
                    asset_weights=self.asset_weights,
                    theme=theme,
                )
            )
        return figs

    def to_figure(self, theme: ThemeName | str = "alloq_dark") -> go.Figure:
        figs = self.get_component_figures(theme=theme)
        if len(figs) == 1:
            return figs[0]

        combined = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Entropy Pooling: Prior vs. Posterior Scenario Profile",
                "2. View-Shifted Return Density (Prior vs. Posterior)",
                "3. Portfolio Equity with Macro Regime Overlay",
                "4. Real-Time Regime Probabilities p_k(t)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        if self.entropy_result is not None:
            fig_ep = plot_entropy_pooling(
                self.entropy_result,
                scenario_data=self.scenario_data,
                asset_weights=self.asset_weights,
                theme=theme,
            )
            for trace in fig_ep.data:
                col_idx = 2 if getattr(trace, "xaxis", "x") in ("x2", "x3") else 1
                combined.add_trace(trace, row=1, col=col_idx)

        if self.equity_or_scenarios is not None and (
            self.gmm_or_probs is not None
            or (
                isinstance(self.equity_or_scenarios, ScenarioMarketData)
                and "regime_states" in self.equity_or_scenarios.features
            )
        ):
            fig_reg = plot_regime_transitions(
                self.equity_or_scenarios,
                gmm_or_probs=self.gmm_or_probs,
                market_data=self.market_data,
                asset_weights=self.asset_weights,
                theme=theme,
            )
            for trace in fig_reg.data:
                col_idx = 2 if getattr(trace, "yaxis", "y") in ("y2", "y3") else 1
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
