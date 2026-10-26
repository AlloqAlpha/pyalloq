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

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba
from pyalloq_viz.utils import compute_kde_1d


def plot_pbo_distribution(
    pbo_source: PBOResult | pd.DataFrame | np.ndarray | Sequence[BacktestResult],
    synthetic_comparison: MonteCarloResult
    | CPCVResult
    | pd.Series
    | np.ndarray
    | None = None,
    n_partitions: int = 8,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Probability of Backtest Overfitting (PBO) & Performance Degradation",
) -> go.Figure:
    """
    Visualizes the Probability of Backtest Overfitting (PBO):
    - In CSCV mode (or when given a PBOResult from CSCV / a (T, M) strategy matrix):
      Left panel plots the distribution of Rank Logits (lambda) with shaded overfitting mass (lambda <= 0).
      Right panel plots In-Sample vs Out-of-Sample Sharpe Ratios with OLS degradation slope.
    - In Single-Strategy Benchmark mode (when `synthetic_comparison` is provided or `PBOResult.mode == 'benchmark'`):
      Left panel plots the simulated/scrambled Sharpe ratio distribution against the strategy's
      historical backtest Sharpe ratio.
      Right panel plots the Empirical Cumulative Distribution (ECDF) and survival probability.
    """
    palette = get_theme(theme)

    if isinstance(pbo_source, PBOResult):
        res = pbo_source
    elif synthetic_comparison is not None:
        evaluator = PBOEvaluator(n_partitions=n_partitions)
        if isinstance(pbo_source, (BacktestResult, pd.Series, float, int)):
            res = evaluator.evaluate_benchmark(pbo_source, synthetic_comparison)
        else:
            res = evaluator.evaluate_cscv(pbo_source)
    else:
        evaluator = PBOEvaluator(n_partitions=n_partitions)
        res = evaluator.evaluate_cscv(pbo_source)

    if res.mode == "cscv":
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Rank Logit (λ) Distribution — PBO = {res.pbo:.1%}",
                f"IS vs. OOS Sharpe Degradation (P[SR_OOS < 0] = {res.prob_oos_loss:.1%})",
            ),
            horizontal_spacing=0.10,
        )

        logits = res.logits
        x_kde, y_kde = compute_kde_1d(logits, n_points=250)

        fig.add_trace(
            go.Histogram(
                x=logits,
                histnorm="probability density",
                nbinsx=min(35, max(10, len(logits) // 3)),
                marker={
                    "color": hex_to_rgba(palette.slate, 0.35),
                    "line": {"color": hex_to_rgba(palette.slate, 0.7), "width": 1},
                },
                name="Logit Histogram",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                line={"color": palette.cyan, "width": 2.5},
                name="Logit Density (KDE)",
            ),
            row=1,
            col=1,
        )

        # Overfit region: lambda <= 0
        overfit_x = x_kde[x_kde <= 0.0]
        overfit_y = y_kde[x_kde <= 0.0]
        if len(overfit_x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=overfit_x,
                    y=overfit_y,
                    mode="lines",
                    line={"color": palette.crimson, "width": 2},
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(palette.crimson, 0.42),
                    name=f"Overfit Mass (λ ≤ 0): {res.pbo:.1%}",
                ),
                row=1,
                col=1,
            )

        fig.add_vline(
            x=0.0,
            line_dash="dash",
            line_color=palette.crimson,
            line_width=2,
            row=1,
            col=1,
        )

        # Panel 2: IS vs OOS Sharpe scatter + regression line
        fig.add_trace(
            go.Scatter(
                x=res.is_sharpes,
                y=res.oos_sharpes,
                mode="markers",
                marker={
                    "size": 8,
                    "color": np.where(
                        res.oos_sharpes >= 0.0, palette.emerald, palette.coral
                    ),
                    "opacity": 0.8,
                    "line": {"width": 0.5, "color": palette.bg_primary},
                },
                name="CSCV Split (IS vs OOS SR)",
                hovertemplate="IS Sharpe: %{x:.2f}<br>OOS Sharpe: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        x_line = np.linspace(
            float(np.min(res.is_sharpes)), float(np.max(res.is_sharpes)), 50
        )
        y_line = res.degradation_slope * x_line + res.degradation_intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line={"color": palette.amber, "width": 2.2, "dash": "dash"},
                name=f"Degradation Fit (Slope={res.degradation_slope:.2f})",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=0.0, line_dash="dot", line_color=palette.border_color, row=1, col=2
        )

        fig.update_xaxes(title_text="Rank Logit (λ_c)", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="In-Sample Sharpe Ratio (SR_IS)", row=1, col=2)
        fig.update_yaxes(title_text="Out-of-Sample Sharpe Ratio (SR_OOS)", row=1, col=2)
        subtitle = (
            f"CSCV Probability of Backtest Overfitting (PBO): {res.pbo:.1%} | "
            f"Prob. of OOS Loss: {res.prob_oos_loss:.1%} | "
            f"Degradation Slope: {res.degradation_slope:.2f}"
        )

    else:
        # Benchmark / Single-Strategy vs Synthetic/CPCV Sharpe distribution
        sim_srs = res.oos_sharpes
        hist_sr = res.historical_sharpe if res.historical_sharpe is not None else 0.0
        pct_rank = float(np.mean(sim_srs <= hist_sr))

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Simulated / Scrambled Sharpe vs. Historical Backtest",
                "Empirical CDF & Strategy Survival Probability",
            ),
            horizontal_spacing=0.10,
        )

        x_kde, y_kde = compute_kde_1d(sim_srs, n_points=250)

        fig.add_trace(
            go.Histogram(
                x=sim_srs,
                histnorm="probability density",
                nbinsx=min(40, max(12, len(sim_srs) // 5)),
                marker={
                    "color": hex_to_rgba(palette.teal, 0.35),
                    "line": {"color": hex_to_rgba(palette.teal, 0.7), "width": 1},
                },
                name="Simulated Sharpe Distribution",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                line={"color": palette.emerald, "width": 2.5},
                name="Sharpe Density (KDE)",
            ),
            row=1,
            col=1,
        )

        # Shade region where Sharpe < 0
        neg_x = x_kde[x_kde <= 0.0]
        neg_y = y_kde[x_kde <= 0.0]
        if len(neg_x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=neg_x,
                    y=neg_y,
                    mode="lines",
                    line={"color": palette.crimson, "width": 2},
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(palette.crimson, 0.4),
                    name=f"Negative Sharpe Mass ({res.prob_oos_loss:.1%})",
                ),
                row=1,
                col=1,
            )

        fig.add_vline(
            x=hist_sr,
            line_dash="solid",
            line_color=palette.amber,
            line_width=3,
            annotation_text=f"Historical SR: {hist_sr:.2f} ({pct_rank:.0%} pct)",
            annotation_position="top right",
            annotation_font_color=palette.amber,
            row=1,
            col=1,
        )

        # Panel 2: Empirical CDF
        sorted_sr = np.sort(sim_srs)
        ecdf_y = np.arange(1, len(sorted_sr) + 1) / len(sorted_sr)
        fig.add_trace(
            go.Scatter(
                x=sorted_sr,
                y=ecdf_y,
                mode="lines",
                line={"color": palette.cyan, "width": 2.5},
                name="Simulated Sharpe ECDF",
            ),
            row=1,
            col=2,
        )
        fig.add_vline(
            x=hist_sr,
            line_dash="dash",
            line_color=palette.amber,
            line_width=2,
            row=1,
            col=2,
        )
        fig.add_hline(
            y=pct_rank,
            line_dash="dot",
            line_color=palette.amber,
            line_width=1.5,
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Annualized Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Annualized Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(
            title_text="Cumulative Probability", tickformat=".0%", row=1, col=2
        )
        subtitle = (
            f"Historical Backtest Sharpe: {hist_sr:.2f} (Percentile: {pct_rank:.1%}) | "
            f"Simulated Mean Sharpe: {float(np.mean(sim_srs)):.2f} | "
            f"Prob(Sharpe < 0): {res.prob_oos_loss:.1%}"
        )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=500)
    return fig


def plot_cpcv_paths(
    cpcv_source: CPCVResult | BacktestResult | pd.Series,
    n_groups: int = 6,
    max_paths_display: int = 50,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Combinatorial Scrambling (CPCV) Timeline Survival",
) -> go.Figure:
    """
    Visualizes strategy survival when historical data blocks are recombined into
    combinatorial timelines, scrambling the chronological order of market regimes.
    Left panel: Combinatorial equity trajectories + median/quantile envelope vs. original historical path.
    Right panel: Distribution of Max Drawdowns vs. Sharpe Ratios across all scrambled regime sequences.
    """
    palette = get_theme(theme)

    if isinstance(cpcv_source, CPCVResult):
        cpcv_res = cpcv_source
    else:
        splitter = CPCVSplitter(n_groups=n_groups)
        cpcv_res = splitter.scramble_timelines(cpcv_source)

    eq_df = cpcv_res.equity_curves
    n_total_paths = eq_df.shape[1]
    x_vals = list(eq_df.index)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        subplot_titles=(
            f"Combinatorial Regime-Scrambled Equity Timelines ({n_total_paths} Paths)",
            "Path Survival: Sharpe vs. Max Drawdown",
        ),
        horizontal_spacing=0.08,
    )

    # Plot up to max_paths_display individual scrambled trajectories
    step = max(1, n_total_paths // max_paths_display)
    for col in eq_df.columns[::step]:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=eq_df[col],
                mode="lines",
                line={"color": hex_to_rgba(palette.violet, 0.18), "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 10th-90th percentile envelope of scrambled timelines
    q10 = eq_df.quantile(0.10, axis=1)
    q90 = eq_df.quantile(0.90, axis=1)
    median_eq = eq_df.median(axis=1)

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=q90,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=q10,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=hex_to_rgba(palette.violet, 0.22),
            name="10th–90th Scrambled Band",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=median_eq,
            mode="lines",
            line={"color": palette.emerald, "width": 2.5},
            name="Median Combinatorial Path",
        ),
        row=1,
        col=1,
    )

    if cpcv_res.historical_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=list(cpcv_res.historical_equity.index),
                y=cpcv_res.historical_equity.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": palette.amber, "width": 2.8},
                name="Chronological Historical Path",
            ),
            row=1,
            col=1,
        )

    # Right panel: Max Drawdown vs Sharpe scatter across scrambled timelines
    mdds = cpcv_res.max_drawdowns.to_numpy(dtype=np.float64)
    srs = cpcv_res.sharpe_ratios.to_numpy(dtype=np.float64)

    fig.add_trace(
        go.Scatter(
            x=mdds,
            y=srs,
            mode="markers",
            marker={
                "size": 8,
                "color": srs,
                "colorscale": "Viridis",
                "showscale": False,
                "opacity": 0.85,
                "line": {"width": 0.5, "color": palette.bg_primary},
            },
            name="Scrambled Realizations",
            hovertemplate="Max Drawdown: %{x:.2%}<br>Sharpe Ratio: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    survival_rate = float(np.mean(srs > 0.0))
    median_mdd = float(np.median(mdds))
    subtitle = (
        f"Regime Blocks: {cpcv_res.n_groups} | "
        f"Survival Rate (SR > 0): {survival_rate:.1%} | "
        f"Median Scrambled Sharpe: {float(np.median(srs)):.2f} | "
        f"Median Scrambled Max DD: {median_mdd:.2%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=500)
    fig.update_xaxes(title_text="Timeline Step / Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Equity (1.0 = Initial)", row=1, col=1)
    fig.update_xaxes(title_text="Max Drawdown (%)", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Annualized Sharpe Ratio", row=1, col=2)
    return fig
