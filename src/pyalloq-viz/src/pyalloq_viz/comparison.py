from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_backtest.attribution import AttributionResult, PerformanceAttributor
from pyalloq_backtest.benchmark import ComparisonResult, MultiStrategyComparator
from pyalloq_backtest.overfitting import CPCVSplitter
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba
from pyalloq_viz.utils import (
    compute_kde_1d,
    extract_equity_paths,
    reconstruct_drifted_weights,
    weighted_quantiles,
)


def _extract_backtest_dict(
    source: ComparisonResult | Mapping[str, BacktestResult | pd.Series],
) -> dict[str, BacktestResult | pd.Series]:
    if isinstance(source, ComparisonResult):
        return dict(source.results)
    return dict(source)


def plot_equity_drawdown_overlay(
    strategy_results: ComparisonResult | Mapping[str, BacktestResult | pd.Series],
    benchmark: BacktestResult | pd.Series | None = None,
    highlight_shock_window: bool = True,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Strategy Cumulative Equity & Underwater Drawdown Overlay",
) -> go.Figure:
    """
    Plots cumulative equity curves overlaid with underwater drawdown charts across multiple
    allocators, isolating which strategies survive correlation breakdowns and volatility spikes.
    """
    palette = get_theme(theme)
    res_map = _extract_backtest_dict(strategy_results)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.07,
        subplot_titles=(
            "Cumulative Net-of-Cost Wealth Trajectories",
            "Underwater Drawdown Profiles (%)",
        ),
    )

    all_dd_series: list[pd.Series] = []
    best_sharpe_name = ""
    best_sharpe_val = -np.inf

    for idx, (name, item) in enumerate(res_map.items()):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        if isinstance(item, BacktestResult):
            ret_s = item.returns.dropna()
            eq = (1.0 + ret_s).cumprod()
            if "sharpe_ratio" in item.tear_sheet.index:
                sr_val = float(item.tear_sheet.loc["sharpe_ratio", "Value"])  # type: ignore[arg-type]
            else:
                std_v = float(ret_s.std() * np.sqrt(252.0))
                sr_val = float(ret_s.mean() * 252.0 / std_v) if std_v > 1e-12 else 0.0
        else:
            s_clean = pd.Series(item).dropna()
            # Detect whether series is already an equity curve (starts near 1.0 and all positive) or returns
            if (
                len(s_clean) > 0
                and float(s_clean.min()) > 0.2
                and abs(float(s_clean.iloc[0]) - 1.0) < 0.25
            ):
                eq = s_clean
                ret_s = eq.pct_change().dropna()
            else:
                ret_s = s_clean
                eq = (1.0 + ret_s).cumprod()
            std_v = float(ret_s.std() * np.sqrt(252.0)) if len(ret_s) > 1 else 0.0
            sr_val = float(ret_s.mean() * 252.0 / std_v) if std_v > 1e-12 else 0.0

        if sr_val > best_sharpe_val:
            best_sharpe_val = sr_val
            best_sharpe_name = name

        x_vals = list(eq.index)
        y_eq = eq.to_numpy(dtype=np.float64)
        rolling_max = eq.cummax().replace(0.0, np.nan)
        dd_s = ((eq - rolling_max) / rolling_max).fillna(0.0)
        all_dd_series.append(dd_s)
        y_dd = dd_s.to_numpy(dtype=np.float64)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_eq,
                mode="lines",
                line={"color": c_hex, "width": 2.4},
                name=name,
                legendgroup=name,
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Equity: %{{y:.3f}}x<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_dd,
                mode="lines",
                line={"color": c_hex, "width": 1.6},
                fill="tozeroy",
                fillcolor=hex_to_rgba(c_hex, 0.14),
                name=f"{name} DD",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if benchmark is not None:
        b_ret = (
            benchmark.returns.dropna()
            if isinstance(benchmark, BacktestResult)
            else pd.Series(benchmark).dropna()
        )
        b_eq = (1.0 + b_ret).cumprod()
        b_max = b_eq.cummax().replace(0.0, np.nan)
        b_dd = ((b_eq - b_max) / b_max).fillna(0.0)
        fig.add_trace(
            go.Scatter(
                x=list(b_eq.index),
                y=b_eq.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": palette.slate, "width": 2.0, "dash": "dash"},
                name="Benchmark",
                legendgroup="Benchmark",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(b_dd.index),
                y=b_dd.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": palette.slate, "width": 1.5, "dash": "dash"},
                name="Benchmark DD",
                legendgroup="Benchmark",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Highlight cross-strategy maximum systemic stress window if dates exist
    if highlight_shock_window and len(all_dd_series) > 0:
        dd_concat = pd.concat(all_dd_series, axis=1).mean(axis=1).dropna()
        if len(dd_concat) > 10 and float(dd_concat.min()) < -0.01:
            trough_idx = int(np.argmin(dd_concat.to_numpy(dtype=np.float64)))
            half_w = max(2, len(dd_concat) // 20)
            start_i = max(0, trough_idx - half_w)
            end_i = min(len(dd_concat) - 1, trough_idx + half_w)
            x0 = dd_concat.index[start_i]
            x1 = dd_concat.index[end_i]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=hex_to_rgba(palette.crimson, 0.10),
                line_width=0,
                annotation_text="Regime Stress Window",
                annotation_position="top left",
            )

    fig.add_hline(y=1.0, line_dash="dot", line_color=palette.border_color, row=1, col=1)
    fig.add_hline(y=0.0, line_color=palette.border_color, line_width=1.2, row=2, col=1)

    subtitle = f"Strategies Compared: {len(res_map)}"
    if best_sharpe_name:
        subtitle += (
            f" | Top Sharpe Strategy: {best_sharpe_name} ({best_sharpe_val:.2f})"
        )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=580)
    fig.update_yaxes(title_text="Cumulative Equity (x)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def plot_multi_strategy_fan_chart(
    source: ComparisonResult
    | Mapping[str, MonteCarloResult | ScenarioMarketData | BacktestResult],
    scenario_data: ScenarioMarketData | None = None,
    percentiles: Sequence[tuple[float, float]] = ((10.0, 90.0), (25.0, 75.0)),
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Strategy Probabilistic Equity Cones & Terminal Wealth Distributions",
) -> go.Figure:
    """
    Renders median equity curves surrounded by shaded confidence intervals (e.g., 10th-90th
    percentile trajectories) across synthetic or scrambled multi-period realities alongside
    overlaid terminal wealth KDE distributions.
    """
    palette = get_theme(theme)

    mc_map: dict[str, tuple[np.ndarray, pd.Index]] = {}
    if isinstance(source, ComparisonResult):
        if source.scenario_results is not None and len(source.scenario_results) > 0:
            for name, mc_res in source.scenario_results.items():
                mc_map[name] = extract_equity_paths(mc_res)
        elif scenario_data is not None:
            for name, bt_res in source.results.items():
                mc_res = MultiStrategyComparator._evaluate_on_scenarios(
                    bt_res, scenario_data
                )
                mc_map[name] = extract_equity_paths(mc_res)
        else:
            splitter = CPCVSplitter(n_groups=6, k_test_groups=2, max_paths=80)
            for name, bt_res in source.results.items():
                cpcv_res = splitter.scramble_timelines(bt_res)
                mc_map[name] = (
                    cpcv_res.equity_curves.to_numpy(dtype=np.float64).T,
                    cpcv_res.equity_curves.index,
                )
    else:
        splitter = CPCVSplitter(n_groups=6, k_test_groups=2, max_paths=80)
        for name, item in source.items():
            if isinstance(item, (MonteCarloResult, ScenarioMarketData)):
                mc_map[name] = extract_equity_paths(item)
            elif isinstance(item, BacktestResult):
                if scenario_data is not None:
                    mc_res = MultiStrategyComparator._evaluate_on_scenarios(
                        item, scenario_data
                    )
                    mc_map[name] = extract_equity_paths(mc_res)
                else:
                    cpcv_res = splitter.scramble_timelines(item)
                    mc_map[name] = (
                        cpcv_res.equity_curves.to_numpy(dtype=np.float64).T,
                        cpcv_res.equity_curves.index,
                    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(
            "Median Equity Trajectories & Shaded Confidence Cones",
            "Terminal Wealth Density (KDE) & Outperformance Probability",
        ),
        horizontal_spacing=0.09,
    )

    sorted_bands = sorted(percentiles, key=lambda b: b[1] - b[0], reverse=True)
    outperf_notes: list[str] = []

    for idx, (name, (eq_paths, timestamps)) in enumerate(mc_map.items()):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        x_vals = list(timestamps)

        for band_i, (low_p, high_p) in enumerate(sorted_bands):
            alpha_fill = 0.12 + 0.08 * band_i
            q_pair = weighted_quantiles(eq_paths, [low_p / 100.0, high_p / 100.0])
            q_low, q_high = q_pair[0], q_pair[1]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=q_high,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=name,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=q_low,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=hex_to_rgba(c_hex, float(alpha_fill)),
                    name=f"{name} ({low_p:.0f}–{high_p:.0f}th Pct)",
                    showlegend=(band_i == 0),
                    legendgroup=name,
                    hovertemplate=f"<b>{name}</b> ({low_p:.0f}–{high_p:.0f}th Band): %{{y:.3f}}x<extra></extra>",
                ),
                row=1,
                col=1,
            )

        median_curve = weighted_quantiles(eq_paths, [0.50])[0]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=median_curve,
                mode="lines",
                line={"color": c_hex, "width": 2.8},
                name=f"{name} (Median)",
                legendgroup=name,
                hovertemplate=f"<b>{name}</b><br>Median Wealth: %{{y:.3f}}x<extra></extra>",
            ),
            row=1,
            col=1,
        )

        term_w = eq_paths[:, -1]
        prob_gain = float(np.mean(term_w > 1.0))
        outperf_notes.append(f"{name}: P(W_T>1)={prob_gain:.0%}")

        x_kde, y_kde = compute_kde_1d(term_w, n_points=200)
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                line={"color": c_hex, "width": 2.2},
                fill="tozeroy",
                fillcolor=hex_to_rgba(c_hex, 0.16),
                name=f"{name} Terminal KDE",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Terminal Wealth: %{{x:.3f}}x<br>Density: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color=palette.border_color, row=1, col=1)
    fig.add_vline(x=1.0, line_dash="dot", line_color=palette.border_color, row=1, col=2)

    subtitle = (
        " | ".join(outperf_notes[:4])
        if outperf_notes
        else "Multi-Strategy Synthetic Cone Comparison"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=520)
    fig.update_xaxes(title_text="Horizon / Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Equity (1.0 = Initial)", row=1, col=1)
    fig.update_xaxes(title_text="Terminal Portfolio Wealth (W_T)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    return fig


def plot_multi_turnover_drift(
    strategy_results: ComparisonResult | Mapping[str, BacktestResult],
    market_data: MarketData | pd.DataFrame | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Strategy Intra-Rebalance Weight Drift (Sawtooth) & Turnover Dynamics",
) -> go.Figure:
    """
    Visualizes daily intra-period price drift between rebalance dates (sawtooth L1 drift patterns)
    alongside two-way rebalance turnover bars and cumulative turnover across allocators.
    """
    palette = get_theme(theme)
    bt_map: dict[str, BacktestResult] = (
        dict(strategy_results.results)
        if isinstance(strategy_results, ComparisonResult)
        else dict(strategy_results)
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.48],
        vertical_spacing=0.08,
        subplot_titles=(
            "Intra-Rebalance Sawtooth Weight Drift (∑ |w_{i,t}^- − w_{i,target}|)",
            "Two-Way Rebalance Turnover Bars & Cumulative Churn (∑ |Δw_{i,t}|)",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    churn_summaries: list[str] = []

    for idx, (name, bt_res) in enumerate(bt_map.items()):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        drifted_df, recon_turnover = reconstruct_drifted_weights(
            bt_res, market_data=market_data
        )

        # Compute daily L1 sawtooth drift relative to the active target weights
        if bt_res.target_weights is not None and not bt_res.target_weights.empty:
            tw = (
                bt_res.target_weights.reindex(
                    index=drifted_df.index, columns=drifted_df.columns
                )
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            sawtooth_l1 = (drifted_df - tw).abs().sum(axis=1)
        else:
            # Measure drift away from most recent rebalance weight snapshot
            reb_mask = recon_turnover > 1e-6
            anchor_df = drifted_df.where(reb_mask).ffill().bfill().fillna(0.0)
            sawtooth_l1 = (drifted_df - anchor_df).abs().sum(axis=1)

        to_series = (
            bt_res.turnover.reindex(drifted_df.index).fillna(0.0)
            if bt_res.turnover is not None
            else recon_turnover
        )
        # Exclude initial t=0 portfolio build from churn comparison if length > 1
        to_post_init = to_series.copy()
        if len(to_post_init) > 1:
            non_zero_idx = np.where(to_post_init.to_numpy(dtype=np.float64) > 1e-6)[0]
            if len(non_zero_idx) > 0:
                to_post_init.iloc[non_zero_idx[0]] = 0.0

        fig.add_trace(
            go.Scatter(
                x=list(sawtooth_l1.index),
                y=sawtooth_l1.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": c_hex, "width": 1.9},
                fill="tozeroy",
                fillcolor=hex_to_rgba(c_hex, 0.12),
                name=f"{name} Drift",
                legendgroup=name,
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>L1 Weight Drift: %{{y:.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        reb_events = to_post_init[to_post_init > 1e-5]
        fig.add_trace(
            go.Bar(
                x=list(reb_events.index),
                y=reb_events.to_numpy(dtype=np.float64),
                marker={"color": c_hex, "opacity": 0.80},
                name=f"{name} Rebalance Turnover",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Rebalance Date: %{{x}}<br>Turnover: %{{y:.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        cum_to = to_post_init.cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(cum_to.index),
                y=cum_to.to_numpy(dtype=np.float64),
                mode="lines",
                line={"color": c_hex, "width": 2.2, "dash": "dot"},
                name=f"{name} Cum. Turnover",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Cum. Turnover: %{{y:.2f}}x<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        total_churn = float(to_post_init.sum())
        churn_summaries.append(f"{name}: {total_churn:.1f}x")

    subtitle = (
        "Post-Init Cumulative Turnover — " + " | ".join(churn_summaries[:5])
        if churn_summaries
        else "Sawtooth Drift & Rebalance Turnover"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=580)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="L1 Drift (∑|w⁻ − w*|)", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(
        title_text="Rebalance Turnover (%)",
        tickformat=".0%",
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Cum. Turnover (x)",
        row=2,
        col=1,
        secondary_y=True,
        showgrid=False,
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def plot_attribution_waterfall(
    source: ComparisonResult
    | AttributionResult
    | Mapping[str, AttributionResult | BacktestResult],
    market_data: MarketData | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Brinson-Fachler Active Decomposition & Frictional Drag Waterfall",
) -> go.Figure:
    """
    Visualizes:
      - Left Panel: Classical Brinson-Fachler Decomposition (Allocation, Selection, Interaction)
        and Factor-Implied Alpha across strategies or sectors.
      - Right Panel: Frictional Drag Accounting Waterfall (Theoretical Paper Return -> Alpha Decay/Drift
        -> Gross Executed -> Spread Cost -> Square-Root Market Impact -> Net Realized Return).
    """
    palette = get_theme(theme)

    attr_map: dict[str, AttributionResult] = {}
    if isinstance(source, AttributionResult):
        attr_map[source.strategy_name] = source
    elif isinstance(source, ComparisonResult):
        if source.attributions:
            attr_map = dict(source.attributions)
        elif market_data is not None:
            attributor = PerformanceAttributor()
            for name, bt_res in source.results.items():
                attr_map[name] = attributor.attribute(bt_res, market_data)
    else:
        attributor = PerformanceAttributor()
        for name, item in source.items():
            if isinstance(item, AttributionResult):
                attr_map[name] = item
            elif isinstance(item, BacktestResult) and market_data is not None:
                attr_map[name] = attributor.attribute(item, market_data)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.50, 0.50],
        subplot_titles=(
            "Brinson-Fachler & Factor Attribution Breakdown",
            "Gross-to-Net Frictional Drag Waterfall (Ma & Smith, 2025)",
        ),
        horizontal_spacing=0.10,
    )

    if len(attr_map) == 1:
        only_attr = next(iter(attr_map.values()))
        sec_df = only_attr.brinson.sector_summary
        x_cats = [str(idx) for idx in sec_df.index]
        fig.add_trace(
            go.Bar(
                x=x_cats,
                y=sec_df["allocation_effect"].to_numpy(dtype=np.float64),
                name="Allocation Effect",
                marker={"color": palette.cyan},
                hovertemplate="<b>%{x}</b><br>Allocation: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x_cats,
                y=sec_df["selection_effect"].to_numpy(dtype=np.float64),
                name="Selection Effect",
                marker={"color": palette.emerald},
                hovertemplate="<b>%{x}</b><br>Selection: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x_cats,
                y=sec_df["interaction_effect"].to_numpy(dtype=np.float64),
                name="Interaction Effect",
                marker={"color": palette.violet},
                hovertemplate="<b>%{x}</b><br>Interaction: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        focal_attr = only_attr
    elif len(attr_map) > 1:
        strat_names = list(attr_map.keys())
        alloc_vals = [attr_map[s].brinson.total_allocation_effect for s in strat_names]
        sel_vals = [attr_map[s].brinson.total_selection_effect for s in strat_names]
        inter_vals = [attr_map[s].brinson.total_interaction_effect for s in strat_names]
        alpha_vals = [attr_map[s].factor.annualized_alpha for s in strat_names]

        fig.add_trace(
            go.Bar(
                x=strat_names,
                y=alloc_vals,
                name="Brinson Allocation",
                marker={"color": palette.cyan},
                hovertemplate="<b>%{x}</b><br>Allocation Effect: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=strat_names,
                y=sel_vals,
                name="Brinson Selection",
                marker={"color": palette.emerald},
                hovertemplate="<b>%{x}</b><br>Selection Effect: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=strat_names,
                y=inter_vals,
                name="Brinson Interaction",
                marker={"color": palette.violet},
                hovertemplate="<b>%{x}</b><br>Interaction Effect: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=strat_names,
                y=alpha_vals,
                mode="markers+lines",
                marker={"size": 10, "symbol": "diamond", "color": palette.amber},
                line={"color": palette.amber, "width": 2, "dash": "dash"},
                name="Factor Alpha (Ann.)",
                hovertemplate="<b>%{x}</b><br>Factor Alpha: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Select strategy with highest net realized return for detailed waterfall
        focal_attr = max(
            attr_map.values(),
            key=lambda a: float(a.frictional.waterfall_summary["net_realized_return"]),
        )
    else:
        focal_attr = None

    if focal_attr is not None:
        wf = focal_attr.frictional.waterfall_summary
        wf_x = [
            "Paper Signal",
            "Drift / Decay",
            "Gross Executed",
            "Spread Cost",
            "Market Impact",
            "Borrow Cost",
            "Net Realized",
        ]
        wf_y = [
            float(wf["theoretical_paper_return"]),
            float(wf["alpha_decay_and_drift_drag"]),
            0.0,
            float(wf["spread_and_commission_drag"]),
            float(wf["market_impact_drag"]),
            float(wf["short_borrow_drag"]),
            0.0,
        ]
        wf_measure = [
            "absolute",
            "relative",
            "total",
            "relative",
            "relative",
            "relative",
            "total",
        ]

        fig.add_trace(
            go.Waterfall(
                name=f"{focal_attr.strategy_name} Frictional Bridge",
                orientation="v",
                measure=wf_measure,
                x=wf_x,
                y=wf_y,
                decreasing={"marker": {"color": palette.coral}},
                increasing={"marker": {"color": palette.emerald}},
                totals={"marker": {"color": palette.cyan}},
                connector={"line": {"color": palette.border_color, "width": 1.5}},
            ),
            row=1,
            col=2,
        )

    subtitle = "Decomposing Active Return (Allocation, Selection, Interaction, Factor Premia) & Execution Frictions"
    if focal_attr is not None:
        subtitle = (
            f"Waterfall Focus: {focal_attr.strategy_name} | "
            f"Factor R²: {focal_attr.factor.r_squared:.1%} | "
            f"Cost Retention: {focal_attr.frictional.cost_retention_ratio:.1%}"
        )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=520)
    fig.update_layout(barmode="group")
    fig.update_yaxes(
        title_text="Active Return / Effect (%)", tickformat=".1%", row=1, col=1
    )
    fig.update_yaxes(
        title_text="Cumulative Return Bridge (%)", tickformat=".1%", row=1, col=2
    )
    return fig
