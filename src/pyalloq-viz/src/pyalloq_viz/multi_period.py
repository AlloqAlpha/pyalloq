from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.linalg as sla
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData
from pyalloq_core.results import BacktestResult, MultiPeriodOptimizationResult

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba


def plot_mpc_trajectory(
    mpo_result: MultiPeriodOptimizationResult | pd.DataFrame,
    current_weights: pd.Series | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Period Planned Weight Trajectory & Stage Trade Execution",
) -> go.Figure:
    """
    Visualizes the forward H-stage receding-horizon plan produced by a multi-period
    optimizer (`MultiPeriodMPCAllocator`, `StochasticScenarioMPCAllocator`, or `IPMOAllocator`):
    - Top Panel: Stacked area ribbon of planned asset weights across stages (t+0 pre-trade
      through t+1 ... t+H).
    - Bottom Panel: Grouped bar chart of planned stage trade vectors z_{tau} = w_{tau} - w_{tau-1}
      overlaid with stage L1 turnover ||z_{tau}||_1.
    """
    palette = get_theme(theme)

    if isinstance(mpo_result, MultiPeriodOptimizationResult):
        if mpo_result.weights_trajectory is None:
            w_traj = pd.DataFrame(
                [mpo_result.weights.to_numpy(dtype=np.float64)],
                index=["t+1"],
                columns=mpo_result.weights.index,
            )
        else:
            w_traj = mpo_result.weights_trajectory.copy()

        if mpo_result.trades_trajectory is not None:
            z_traj = mpo_result.trades_trajectory.copy()
        else:
            w_vals = w_traj.to_numpy(dtype=np.float64)
            if current_weights is not None:
                w0 = (
                    current_weights.reindex(w_traj.columns)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                w0 = w_vals[0]
            diffs = np.vstack([w_vals[0] - w0, np.diff(w_vals, axis=0)])
            z_traj = pd.DataFrame(diffs, index=w_traj.index, columns=w_traj.columns)
    else:
        w_traj = mpo_result.copy()
        w_vals = w_traj.to_numpy(dtype=np.float64)
        if current_weights is not None:
            w0 = (
                current_weights.reindex(w_traj.columns)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            w0 = w_vals[0]
        diffs = np.vstack([w_vals[0] - w0, np.diff(w_vals, axis=0)])
        z_traj = pd.DataFrame(diffs, index=w_traj.index, columns=w_traj.columns)

    # Prepend current_weights (t+0 pre-trade state) to weight trajectory if provided or inferable
    if current_weights is not None:
        w0_row = pd.DataFrame(
            [
                current_weights.reindex(w_traj.columns)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            ],
            index=["t+0 (Pre-Trade)"],
            columns=w_traj.columns,
        )
        w_plot_df = pd.concat([w0_row, w_traj], axis=0)
    elif isinstance(mpo_result, MultiPeriodOptimizationResult) and (
        mpo_result.trades_trajectory is not None
    ):
        inferred_w0 = w_traj.iloc[0].to_numpy(dtype=np.float64) - z_traj.iloc[
            0
        ].to_numpy(dtype=np.float64)
        w0_row = pd.DataFrame(
            [inferred_w0],
            index=["t+0 (Pre-Trade)"],
            columns=w_traj.columns,
        )
        w_plot_df = pd.concat([w0_row, w_traj], axis=0)
    else:
        w_plot_df = w_traj

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.12,
        subplot_titles=(
            "Planned Multi-Stage Portfolio Weight Trajectory (w_τ)",
            "Planned Child-Order Trade Splitting (z_τ = w_τ − w_{τ−1}) & Stage Turnover",
        ),
    )

    x_w = [str(idx) for idx in w_plot_df.index]
    for idx, asset in enumerate(w_plot_df.columns):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        fig.add_trace(
            go.Scatter(
                x=x_w,
                y=w_plot_df[asset].to_numpy(dtype=np.float64),
                mode="lines+markers",
                stackgroup="mpc_weights",
                line={"color": c_hex, "width": 1.5},
                fillcolor=hex_to_rgba(c_hex, 0.72),
                name=str(asset),
                hovertemplate=f"<b>{asset}</b> (Stage %{{x}}): %{{y:.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    x_z = [str(idx) for idx in z_traj.index]
    for idx, asset in enumerate(z_traj.columns):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        fig.add_trace(
            go.Bar(
                x=x_z,
                y=z_traj[asset].to_numpy(dtype=np.float64),
                marker={"color": c_hex},
                name=f"Δ {asset}",
                showlegend=False,
                hovertemplate=f"<b>Δ {asset}</b> (Stage %{{x}}): %{{y:+.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    stage_turnover = np.sum(np.abs(z_traj.to_numpy(dtype=np.float64)), axis=1)
    fig.add_trace(
        go.Scatter(
            x=x_z,
            y=stage_turnover,
            mode="lines+markers",
            line={"color": palette.amber, "width": 2.5, "dash": "dash"},
            marker={"size": 8, "symbol": "diamond", "color": palette.amber},
            name="Stage L1 Turnover (∑|z_τ|)",
            hovertemplate="Stage %{x}<br>L1 Turnover: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    first_step_to = float(stage_turnover[0]) if len(stage_turnover) > 0 else 0.0
    total_to = float(np.sum(stage_turnover))
    subtitle = (
        f"Horizon Stages: {len(z_traj)} | "
        f"Immediate Execution Turnover (t+1): {first_step_to:.2%} | "
        f"Total Planned Horizon Turnover: {total_to:.2%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=620)
    fig.update_layout(barmode="group")
    fig.update_yaxes(
        title_text="Portfolio Weight", tickformat=".0%", range=[0.0, 1.0], row=1, col=1
    )
    fig.update_yaxes(title_text="Trade / Turnover (%)", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Planning Horizon Stage", row=2, col=1)
    return fig


def plot_alpha_decay_surface(
    horizon_forecasts: pd.DataFrame,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Horizon Alpha Decay Term Structure (Gârleanu-Pedersen)",
) -> go.Figure:
    """
    Visualizes the (H x N) multi-stage expected return matrix produced by
    `AlphaDecayReturnEstimator.estimate_horizon(data)`:
    - Left Panel: Decay curves of forecasted expected returns across horizon stages t+1 ... t+H.
    - Right Panel: Horizon x Asset heatmap of expected return intensity.
    """
    palette = get_theme(theme)
    stages = [str(idx) for idx in horizon_forecasts.index]
    assets = [str(col) for col in horizon_forecasts.columns]
    vals = horizon_forecasts.to_numpy(dtype=np.float64)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.54, 0.46],
        subplot_titles=(
            "Expected Return Decay Trajectories Across Horizon",
            "Horizon × Asset Alpha Intensity Heatmap",
        ),
        horizontal_spacing=0.10,
    )

    for idx, asset in enumerate(assets):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        fig.add_trace(
            go.Scatter(
                x=stages,
                y=vals[:, idx],
                mode="lines+markers",
                line={"color": c_hex, "width": 2.5},
                marker={"size": 7},
                name=asset,
                hovertemplate=f"<b>{asset}</b> (Stage %{{x}}): %{{y:.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=vals.T,
            x=stages,
            y=assets,
            colorscale="Tealgrn",
            text=np.round(vals.T * 100.0, 2),
            texttemplate="%{text:.2f}%",
            showscale=True,
            colorbar={"title": "E[r]", "thickness": 12, "len": 0.75},
            hovertemplate="Asset: <b>%{y}</b><br>Stage: %{x}<br>Forecast: %{z:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    init_norm = float(np.linalg.norm(vals[0])) if len(vals) > 0 else 0.0
    term_norm = float(np.linalg.norm(vals[-1])) if len(vals) > 0 else 0.0
    retention = (term_norm / init_norm) if init_norm > 1e-12 else 1.0
    subtitle = (
        f"Horizon Length: {len(stages)} stages | "
        f"Assets: {len(assets)} | "
        f"Terminal Signal Retention (||μ_H|| / ||μ_1||): {retention:.1%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=480)
    fig.update_xaxes(title_text="Horizon Stage", row=1, col=1)
    fig.update_yaxes(
        title_text="Expected Return Forecast", tickformat=".1%", row=1, col=1
    )
    fig.update_xaxes(title_text="Horizon Stage", row=1, col=2)
    fig.update_yaxes(title_text="Asset", row=1, col=2)
    return fig


def _compute_rolling_thermodynamics(
    market_data_or_returns: MarketData | pd.DataFrame,
    fast_window: int = 15,
    slow_window: int = 45,
    w2_threshold: float = 0.025,
    sensitivity: float = 25.0,
) -> pd.DataFrame:
    if isinstance(market_data_or_returns, MarketData):
        rets_df = market_data_or_returns.prices.pct_change().fillna(0.0)
    else:
        rets_df = market_data_or_returns.copy().fillna(0.0)

    n_obs, n_assets = rets_df.shape
    eps_eye = np.eye(n_assets, dtype=float) * 1e-8
    records: list[dict[str, float]] = []
    dates = rets_df.index

    for t in range(n_obs):
        if t < max(5, fast_window):
            records.append(
                {
                    "wasserstein_distance": 0.0,
                    "temp_fast": 1e-4,
                    "temp_slow": 1e-4,
                    "carnot_efficiency": 0.0,
                    "damping_factor": 1.0,
                }
            )
            continue

        fast_slice = rets_df.iloc[max(0, t - fast_window + 1) : t + 1]
        slow_slice = rets_df.iloc[max(0, t - slow_window + 1) : t + 1]

        mu_f = fast_slice.mean().to_numpy(dtype=float)
        cov_f = fast_slice.cov().fillna(0.0).to_numpy(dtype=float)
        mu_s = slow_slice.mean().to_numpy(dtype=float)
        cov_s = slow_slice.cov().fillna(0.0).to_numpy(dtype=float)

        c1 = 0.5 * (cov_f + cov_f.T) + eps_eye
        c2 = 0.5 * (cov_s + cov_s.T) + eps_eye
        mean_diff_sq = float(np.sum((mu_f - mu_s) ** 2))
        sqrt_c1 = np.real(sla.sqrtm(c1))
        inner = 0.5 * (sqrt_c1 @ c2 @ sqrt_c1 + (sqrt_c1 @ c2 @ sqrt_c1).T) + eps_eye
        sqrt_inner = np.real(sla.sqrtm(inner))
        bures_sq = float(np.trace(c1 + c2 - 2.0 * sqrt_inner))
        w2 = float(np.sqrt(max(0.0, mean_diff_sq + bures_sq)))

        t_f = max(float(np.trace(cov_f)) / max(1, n_assets), 1e-10)
        t_s = max(float(np.trace(cov_s)) / max(1, n_assets), 1e-10)
        carnot = float(np.clip(1.0 - (min(t_f, t_s) / max(t_f, t_s)), 0.0, 1.0))
        excess = max(0.0, w2 - w2_threshold)
        damping = float(np.clip(np.exp(-sensitivity * excess), 0.0, 1.0))

        records.append(
            {
                "wasserstein_distance": w2,
                "temp_fast": t_f,
                "temp_slow": t_s,
                "carnot_efficiency": carnot,
                "damping_factor": damping,
            }
        )

    return pd.DataFrame(records, index=dates)


def plot_thermodynamic_regime_gate(
    market_data_or_diagnostics: MarketData | pd.DataFrame,
    fast_window: int = 15,
    slow_window: int = 45,
    w2_threshold: float = 0.025,
    sensitivity: float = 25.0,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Thermodynamic & Bures-Wasserstein Regime Gate Diagnostics",
) -> go.Figure:
    """
    Visualizes the Physics-Informed Non-Equilibrium Regime Gate (Moroke et al., 2026):
    - Panel 1: Rolling Bures-Wasserstein W_2(t) distance between fast and slow market covariance
      baths relative to the critical gating threshold W_{2,thresh}.
    - Panel 2: Carnot Efficiency Bound eta_{Carnot}(t) = 1 - T_min / T_max.
    - Panel 3: Rebalance Trade Damping Factor lambda_{gate}(t) in [0, 1].
    """
    palette = get_theme(theme)

    if isinstance(market_data_or_diagnostics, pd.DataFrame) and {
        "wasserstein_distance",
        "carnot_efficiency",
        "damping_factor",
    }.issubset(set(market_data_or_diagnostics.columns)):
        diag_df = market_data_or_diagnostics.copy()
    else:
        diag_df = _compute_rolling_thermodynamics(
            market_data_or_diagnostics,
            fast_window=fast_window,
            slow_window=slow_window,
            w2_threshold=w2_threshold,
            sensitivity=sensitivity,
        )

    x_vals = list(diag_df.index)
    w2_vals = diag_df["wasserstein_distance"].to_numpy(dtype=np.float64)
    carnot_vals = diag_df["carnot_efficiency"].to_numpy(dtype=np.float64)
    damping_vals = diag_df["damping_factor"].to_numpy(dtype=np.float64)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.31, 0.31],
        vertical_spacing=0.07,
        subplot_titles=(
            "Bures-Wasserstein Regime Distance W₂(𝒩_fast, 𝒩_slow) vs. Critical Threshold",
            "Thermodynamic Carnot Efficiency Bound (η_Carnot = 1 − T_min / T_max)",
            "Rebalance Trade Damping Factor λ_gate(t) ∈ [0, 1]",
        ),
    )

    # Panel 1: W2 distance + threshold line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=w2_vals,
            mode="lines",
            line={"color": palette.coral, "width": 2.0},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.coral, 0.18),
            name="Bures-Wasserstein W₂",
            hovertemplate="Date: %{x}<br>W₂ Distance: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=np.full(len(x_vals), w2_threshold),
            mode="lines",
            line={"color": palette.amber, "width": 1.8, "dash": "dash"},
            name=f"Gating Threshold ({w2_threshold:.3f})",
            hovertemplate="Threshold: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Carnot efficiency
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=carnot_vals,
            mode="lines",
            line={"color": palette.violet, "width": 2.0},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.violet, 0.18),
            name="Carnot Efficiency (η_Carnot)",
            hovertemplate="Date: %{x}<br>η_Carnot: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Panel 3: Trade Damping Factor
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=damping_vals,
            mode="lines",
            line={"color": palette.emerald, "width": 2.2},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.emerald, 0.22),
            name="Trade Execution Scale (λ_gate)",
            hovertemplate="Date: %{x}<br>λ_gate: %{y:.1%}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    gated_pct = float((damping_vals < 0.99).mean()) if len(damping_vals) > 0 else 0.0
    subtitle = (
        f"Fast/Slow Baths: {fast_window}d / {slow_window}d | "
        f"Peak W₂: {float(np.max(w2_vals)):.4f} | "
        f"Mean Carnot Efficiency: {float(np.mean(carnot_vals)):.1%} | "
        f"Regime-Damped Days: {gated_pct:.1%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=640)
    fig.update_yaxes(title_text="W₂ Distance", row=1, col=1)
    fig.update_yaxes(
        title_text="η_Carnot", tickformat=".0%", range=[0.0, 1.05], row=2, col=1
    )
    fig.update_yaxes(
        title_text="λ_gate", tickformat=".0%", range=[0.0, 1.05], row=3, col=1
    )
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def plot_mpo_pillar_comparison(
    strategy_results: Mapping[str, BacktestResult],
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Period Optimization (3-Pillar) Net-of-Cost Benchmark",
) -> go.Figure:
    """
    Compares multiple `BacktestResult` runs across the 3 MPO Pillars:
    - Panel 1: Cumulative Net-of-Cost Equity Curves.
    - Panel 2: Underwater Drawdown Profiles.
    - Panel 3: Cumulative Realized Two-Way Turnover (∑ ||Δw_t||_1), highlighting
      how multi-period optimization suppresses frictional trading drag.
    """
    palette = get_theme(theme)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.29, 0.29],
        vertical_spacing=0.07,
        subplot_titles=(
            "Cumulative Net-of-Cost Wealth Growth",
            "Underwater Drawdown Profile (%)",
            "Cumulative Realized Portfolio Turnover (∑ ||Δw_t||₁)",
        ),
    )

    for idx, (name, bt_res) in enumerate(strategy_results.items()):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        eq = bt_res.equity_curve.dropna()
        x_eq = list(eq.index)
        y_eq = eq.to_numpy(dtype=np.float64)

        rolling_max = eq.cummax().replace(0.0, np.nan)
        dd = ((eq - rolling_max) / rolling_max).fillna(0.0).to_numpy(dtype=np.float64)

        if bt_res.turnover is not None:
            cum_to = bt_res.turnover.reindex(eq.index).fillna(0.0).cumsum()
            y_to = cum_to.to_numpy(dtype=np.float64)
        else:
            y_to = np.zeros_like(y_eq)

        fig.add_trace(
            go.Scatter(
                x=x_eq,
                y=y_eq,
                mode="lines",
                line={"color": c_hex, "width": 2.2},
                name=name,
                legendgroup=name,
                hovertemplate=f"<b>{name}</b><br>Equity: %{{y:.3f}}x<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_eq,
                y=dd,
                mode="lines",
                line={"color": c_hex, "width": 1.6},
                name=name,
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Drawdown: %{{y:.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_eq,
                y=y_to,
                mode="lines",
                line={"color": c_hex, "width": 2.0},
                name=name,
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Cum. Turnover: %{{y:.2f}}x<extra></extra>",
            ),
            row=3,
            col=1,
        )

    subtitle = f"Strategies Evaluated: {len(strategy_results)} | Net of Transaction Costs & Intra-Period Weight Drift"
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=700)
    fig.update_yaxes(title_text="Wealth (x)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
    fig.update_yaxes(title_text="Cum. Turnover (x)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig
