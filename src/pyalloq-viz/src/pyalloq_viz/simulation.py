from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyalloq_core.data import ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult
from pyalloq_simulation.views.pooler import EntropyResult

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba
from pyalloq_viz.utils import (
    compute_drawdown_paths,
    compute_kde_1d,
    extract_equity_paths,
    weighted_quantiles,
)


def plot_fan_chart(
    source: MonteCarloResult | ScenarioMarketData | pd.DataFrame | np.ndarray,
    asset_weights: np.ndarray | pd.Series | None = None,
    entropy_result: EntropyResult | None = None,
    percentiles: Sequence[tuple[float, float]] = (
        (5.0, 95.0),
        (10.0, 90.0),
        (25.0, 75.0),
    ),
    benchmark: BacktestResult | pd.Series | None = None,
    n_sample_paths: int = 15,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Probabilistic Equity Cone (Fan Chart)",
) -> go.Figure:
    """
    Plots the median equity trajectory enveloped by shaded confidence intervals
    (e.g., 5th-95th, 10th-90th, 25th-75th percentiles) across synthetic futures.
    Supports probability-weighted quantiles when `entropy_result` is provided.
    """
    palette = get_theme(theme)
    equity_paths, timestamps = extract_equity_paths(source, asset_weights=asset_weights)
    M, T = equity_paths.shape
    x_vals = list(timestamps)

    probs = (
        np.asarray(entropy_result.weights, dtype=np.float64)
        if entropy_result is not None
        else None
    )

    fig = go.Figure()

    # Optional faint sample spaghetti trajectories
    if n_sample_paths > 0 and M > 0:
        n_draw = min(n_sample_paths, M)
        sample_indices = np.linspace(0, M - 1, n_draw, dtype=int)
        for idx in sample_indices:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=equity_paths[idx],
                    mode="lines",
                    line={"color": hex_to_rgba(palette.slate, 0.16), "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"Path {idx}",
                )
            )

    # Sort percentile bands from widest to narrowest
    sorted_bands = sorted(percentiles, key=lambda b: b[1] - b[0], reverse=True)
    alphas = np.linspace(0.14, 0.32, max(1, len(sorted_bands)))

    for (low_p, high_p), alpha_val in zip(sorted_bands, alphas):
        q_pair = weighted_quantiles(
            equity_paths, [low_p / 100.0, high_p / 100.0], sample_weight=probs
        )
        q_low, q_high = q_pair[0], q_pair[1]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q_high,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name=f"{high_p:.0f}th Percentile",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q_low,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=hex_to_rgba(palette.emerald, float(alpha_val)),
                name=f"{low_p:.0f}th–{high_p:.0f}th Percentile Band",
                hovertemplate=f"{low_p:.0f}th–{high_p:.0f}th Band: %{{y:.3f}}<extra></extra>",
            )
        )

    # Median & Mean trajectories
    median_curve = weighted_quantiles(equity_paths, [0.50], sample_weight=probs)[0]
    mean_curve = (
        np.average(equity_paths, axis=0, weights=probs)
        if probs is not None
        else np.mean(equity_paths, axis=0)
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=median_curve,
            mode="lines",
            line={"color": palette.emerald, "width": 3},
            name="Median Trajectory (50th Pct)",
            hovertemplate="Median Wealth: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=mean_curve,
            mode="lines",
            line={"color": palette.cyan, "width": 2, "dash": "dash"},
            name="Expected (Mean) Trajectory",
            hovertemplate="Expected Wealth: %{y:.4f}<extra></extra>",
        )
    )

    if benchmark is not None:
        bench_eq = (
            benchmark.equity_curve
            if isinstance(benchmark, BacktestResult)
            else pd.Series(benchmark)
        )
        fig.add_trace(
            go.Scatter(
                x=list(bench_eq.index),
                y=bench_eq.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": palette.amber, "width": 2.2},
                name="Realized / Benchmark Equity",
            )
        )

    # Initial wealth reference line
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=palette.border_color,
        annotation_text="Initial Capital (1.0)",
        annotation_position="bottom right",
    )

    subtitle = f"Dispersion across {M:,} synthetic paths over {T} horizon steps"
    if entropy_result is not None:
        subtitle += f" (View-Adjusted Posterior, KL={entropy_result.kl_divergence:.4f})"

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle)
    fig.update_xaxes(title_text="Simulation Horizon / Date")
    fig.update_yaxes(title_text="Portfolio Equity (Normalized = 1.0)")
    return fig


def plot_terminal_wealth_ruin(
    source: MonteCarloResult | ScenarioMarketData | pd.DataFrame | np.ndarray,
    asset_weights: np.ndarray | pd.Series | None = None,
    entropy_result: EntropyResult | None = None,
    ruin_threshold: float = -0.20,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Terminal Wealth Distribution & Probability of Ruin",
) -> go.Figure:
    """
    Plots the density distribution of the portfolio's terminal wealth, isolating
    the probability mass falling below the ruin threshold (1 + ruin_threshold).
    Also reports path-dependent intra-horizon ruin probability.
    """
    palette = get_theme(theme)
    equity_paths, _ = extract_equity_paths(source, asset_weights=asset_weights)
    M, _ = equity_paths.shape
    terminal_wealth = equity_paths[:, -1]

    probs = (
        np.asarray(entropy_result.weights, dtype=np.float64)
        if entropy_result is not None
        else np.full(M, 1.0 / M, dtype=np.float64)
    )
    probs = probs / np.sum(probs)

    # Normalize ruin_threshold: if negative (e.g. -0.20), terminal ruin wealth is 1.0 - 0.20 = 0.80
    ruin_drop = -abs(ruin_threshold)
    wealth_ruin_level = 1.0 + ruin_drop

    # Terminal ruin probability & path-dependent intra-horizon ruin probability
    terminal_ruin_mask = terminal_wealth <= wealth_ruin_level
    prob_terminal_ruin = float(np.dot(probs, terminal_ruin_mask.astype(np.float64)))

    drawdowns = compute_drawdown_paths(equity_paths)
    path_mdds = np.min(drawdowns, axis=1)
    min_wealth = np.min(equity_paths, axis=1)
    intra_ruin_mask = (path_mdds <= ruin_drop) | (min_wealth <= wealth_ruin_level)
    prob_path_ruin = float(np.dot(probs, intra_ruin_mask.astype(np.float64)))

    expected_wealth = float(np.dot(probs, terminal_wealth))

    x_kde, y_kde = compute_kde_1d(terminal_wealth, weights=probs, n_points=300)

    fig = go.Figure()

    # Histogram of terminal wealth
    fig.add_trace(
        go.Histogram(
            x=terminal_wealth,
            histnorm="probability density",
            nbinsx=min(50, max(15, M // 8)),
            marker={
                "color": hex_to_rgba(palette.teal, 0.35),
                "line": {"color": hex_to_rgba(palette.teal, 0.7), "width": 1},
            },
            name="Terminal Wealth Histogram",
            hovertemplate="Terminal Wealth: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
        )
    )

    # Full KDE curve
    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode="lines",
            line={"color": palette.emerald, "width": 2.5},
            name="Wealth Density (KDE)",
        )
    )

    # Shaded ruin tail (x <= wealth_ruin_level)
    ruin_x = x_kde[x_kde <= wealth_ruin_level]
    ruin_y = y_kde[x_kde <= wealth_ruin_level]
    if len(ruin_x) > 1:
        fig.add_trace(
            go.Scatter(
                x=ruin_x,
                y=ruin_y,
                mode="lines",
                line={"color": palette.crimson, "width": 2},
                fill="tozeroy",
                fillcolor=hex_to_rgba(palette.crimson, 0.45),
                name=f"Ruin Tail (≤ {wealth_ruin_level:.2f})",
            )
        )

    # Vertical threshold lines
    fig.add_vline(
        x=wealth_ruin_level,
        line_dash="dash",
        line_color=palette.crimson,
        line_width=2.5,
        annotation_text=f"Ruin Threshold ({wealth_ruin_level:.2f}) | Terminal Ruin: {prob_terminal_ruin:.1%}",
        annotation_position="top left",
        annotation_font_color=palette.crimson,
    )
    fig.add_vline(
        x=expected_wealth,
        line_dash="dot",
        line_color=palette.cyan,
        line_width=2,
        annotation_text=f"E[W_T] = {expected_wealth:.3f}",
        annotation_position="top right",
        annotation_font_color=palette.cyan,
    )

    subtitle = (
        f"Terminal Ruin Prob: {prob_terminal_ruin:.2%} | "
        f"Intra-Horizon Ruin Prob (DD ≤ {ruin_drop:.0%}): {prob_path_ruin:.2%} | "
        f"Expected Terminal Wealth: {expected_wealth:.3f}"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle)
    fig.update_xaxes(title_text="Terminal Portfolio Wealth (W_T)")
    fig.update_yaxes(title_text="Probability Density")
    return fig


def plot_cvar_tail_risk(
    source: MonteCarloResult
    | ScenarioMarketData
    | BacktestResult
    | pd.DataFrame
    | pd.Series
    | np.ndarray,
    asset_weights: np.ndarray | pd.Series | None = None,
    entropy_result: EntropyResult | None = None,
    alpha: float = 0.05,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Return Distribution & CVaR Tail-Risk Shading",
) -> go.Figure:
    """
    On the return distribution histogram, visually isolates the left tail to demonstrate
    Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall) during
    extreme market crashes.
    """
    palette = get_theme(theme)

    if isinstance(source, BacktestResult):
        returns_1d = source.returns.dropna().to_numpy(dtype=np.float64)
        probs = np.full(len(returns_1d), 1.0 / max(1, len(returns_1d)))
        horizon_label = "Daily / Period Portfolio Return"
    elif isinstance(source, pd.Series):
        returns_1d = source.dropna().to_numpy(dtype=np.float64)
        probs = np.full(len(returns_1d), 1.0 / max(1, len(returns_1d)))
        horizon_label = "Portfolio Return"
    elif isinstance(source, np.ndarray) and source.ndim == 1:
        returns_1d = source.astype(np.float64)
        probs = np.full(len(returns_1d), 1.0 / max(1, len(returns_1d)))
        horizon_label = "Portfolio Return"
    else:
        equity_paths, _ = extract_equity_paths(source, asset_weights=asset_weights)
        returns_1d = equity_paths[:, -1] - 1.0
        probs = (
            np.asarray(entropy_result.weights, dtype=np.float64)
            if entropy_result is not None
            else np.full(len(returns_1d), 1.0 / max(1, len(returns_1d)))
        )
        horizon_label = "Cumulative Horizon Return"

    probs = probs / np.sum(probs)

    # Calculate probability-weighted VaR and CVaR
    sort_idx = np.argsort(returns_1d)
    sorted_rets = returns_1d[sort_idx]
    sorted_probs = probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)

    var_alpha = float(weighted_quantiles(returns_1d, [alpha], sample_weight=probs)[0])
    tail_idx = np.where(cum_probs <= alpha)[0]
    if len(tail_idx) == 0:
        cvar_alpha = float(sorted_rets[0])
    else:
        tail_p = sorted_probs[tail_idx] / np.sum(sorted_probs[tail_idx])
        cvar_alpha = float(np.dot(sorted_rets[tail_idx], tail_p))

    mean_ret = float(np.dot(returns_1d, probs))
    conf_pct = round((1.0 - alpha) * 100)

    x_kde, y_kde = compute_kde_1d(returns_1d, weights=probs, n_points=300)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns_1d,
            histnorm="probability density",
            nbinsx=min(55, max(15, len(returns_1d) // 8)),
            marker={
                "color": hex_to_rgba(palette.slate, 0.35),
                "line": {"color": hex_to_rgba(palette.slate, 0.65), "width": 1},
            },
            name="Return Distribution",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode="lines",
            line={"color": palette.cyan, "width": 2.5},
            name="Return Density (KDE)",
        )
    )

    # Shade the left tail <= VaR_alpha
    tail_x = x_kde[x_kde <= var_alpha]
    tail_y = y_kde[x_kde <= var_alpha]
    if len(tail_x) == 0:
        tail_x = np.array([np.min(x_kde), var_alpha])
        tail_y = np.interp(tail_x, x_kde, y_kde)

    fig.add_trace(
        go.Scatter(
            x=tail_x,
            y=tail_y,
            mode="lines",
            line={"color": palette.coral, "width": 2},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.coral, 0.45),
            name=f"{conf_pct}% Tail Shortfall Region (≤ VaR)",
        )
    )

    fig.add_vline(
        x=var_alpha,
        line_dash="dash",
        line_color=palette.amber,
        line_width=2,
        annotation_text=f"VaR {conf_pct}%: {var_alpha:.2%}",
        annotation_position="top right",
        annotation_font_color=palette.amber,
    )
    fig.add_vline(
        x=cvar_alpha,
        line_dash="solid",
        line_color=palette.crimson,
        line_width=2.8,
        annotation_text=f"CVaR {conf_pct}% (Expected Shortfall): {cvar_alpha:.2%}",
        annotation_position="top left",
        annotation_font_color=palette.crimson,
    )

    subtitle = (
        f"Mean Return: {mean_ret:.2%} | "
        f"VaR ({conf_pct}%): {var_alpha:.2%} | "
        f"CVaR ({conf_pct}%): {cvar_alpha:.2%}"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle)
    fig.update_xaxes(title_text=horizon_label, tickformat=".1%")
    fig.update_yaxes(title_text="Probability Density")
    return fig


def plot_drawdown_fan_chart(
    source: MonteCarloResult | ScenarioMarketData | pd.DataFrame | np.ndarray,
    asset_weights: np.ndarray | pd.Series | None = None,
    entropy_result: EntropyResult | None = None,
    percentiles: Sequence[tuple[float, float]] = (
        (5.0, 95.0),
        (10.0, 90.0),
        (25.0, 75.0),
    ),
    theme: ThemeName | str = "alloq_dark",
    title: str = "Underwater Drawdown Fan Chart",
) -> go.Figure:
    """
    Plots an underwater area chart of the probability distribution of drawdowns across
    all synthetic paths, illustrating loss severity and duration across the multiverse.
    """
    palette = get_theme(theme)
    equity_paths, timestamps = extract_equity_paths(source, asset_weights=asset_weights)
    drawdown_paths = compute_drawdown_paths(equity_paths)
    x_vals = list(timestamps)

    probs = (
        np.asarray(entropy_result.weights, dtype=np.float64)
        if entropy_result is not None
        else None
    )

    fig = go.Figure()

    sorted_bands = sorted(percentiles, key=lambda b: b[1] - b[0], reverse=True)
    alphas = np.linspace(0.14, 0.34, max(1, len(sorted_bands)))

    for (low_p, high_p), alpha_val in zip(sorted_bands, alphas):
        q_pair = weighted_quantiles(
            drawdown_paths, [low_p / 100.0, high_p / 100.0], sample_weight=probs
        )
        q_worst, q_mild = q_pair[0], q_pair[1]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q_mild,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q_worst,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=hex_to_rgba(palette.coral, float(alpha_val)),
                name=f"{low_p:.0f}th–{high_p:.0f}th Drawdown Band",
                hovertemplate=f"{low_p:.0f}th–{high_p:.0f}th DD: %{{y:.2%}}<extra></extra>",
            )
        )

    median_dd = weighted_quantiles(drawdown_paths, [0.50], sample_weight=probs)[0]
    worst_5pct_dd = weighted_quantiles(drawdown_paths, [0.05], sample_weight=probs)[0]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=median_dd,
            mode="lines",
            line={"color": palette.coral, "width": 2.8},
            name="Median Drawdown (50th Pct)",
            hovertemplate="Median Drawdown: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=worst_5pct_dd,
            mode="lines",
            line={"color": palette.crimson, "width": 2, "dash": "dash"},
            name="Tail Drawdown (5th Pct)",
            hovertemplate="5th Pct Drawdown: %{y:.2%}<extra></extra>",
        )
    )

    path_mdds = np.min(drawdown_paths, axis=1)
    expected_mdd = (
        float(np.dot(probs / np.sum(probs), path_mdds))
        if probs is not None
        else float(np.mean(path_mdds))
    )

    fig.add_hline(y=0.0, line_color=palette.border_color, line_width=1.5)

    subtitle = (
        f"Expected Max Drawdown across paths: {expected_mdd:.2%} | "
        f"95th Percentile Worst Drawdown: {float(np.min(worst_5pct_dd)):.2%}"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle)
    fig.update_xaxes(title_text="Simulation Horizon / Date")
    fig.update_yaxes(title_text="Underwater Drawdown (%)", tickformat=".1%")
    return fig
