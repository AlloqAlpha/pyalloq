from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult
from pyalloq_simulation.generators.regimes import GMMRegimeGenerator
from pyalloq_simulation.views.pooler import EntropyResult

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba
from pyalloq_viz.utils import compute_kde_1d, extract_equity_paths, weighted_quantiles


def plot_entropy_pooling(
    entropy_result: EntropyResult,
    scenario_data: ScenarioMarketData | np.ndarray | pd.Series | None = None,
    asset_weights: np.ndarray | pd.Series | None = None,
    sort_by_posterior: bool = True,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Entropy Pooling: Prior vs. Posterior Probability & Return Density",
) -> go.Figure:
    """
    Two-panel visualization of Meucci's Entropy Pooling:
    - Panel 1 (Left): Comparative probability vector profile displaying the original
      equal-weight prior probabilities p against the solved posterior probabilities x*,
      annotated with KL-divergence and Effective Number of Scenarios (ENS).
    - Panel 2 (Right): Prior vs. Posterior probability-weighted return density (when
      `scenario_data` or scenario returns are supplied) or Lorenz probability concentration curve.
    """
    palette = get_theme(theme)
    prior = np.asarray(entropy_result.prior, dtype=np.float64)
    posterior = np.asarray(entropy_result.weights, dtype=np.float64)
    M = len(posterior)

    # Effective Number of Scenarios: ENS = exp(-sum(x_i * ln(x_i)))
    safe_post = np.where(posterior > 1e-16, posterior, 1e-16)
    entropy_val = -float(np.sum(posterior * np.log(safe_post)))
    ens = float(np.exp(entropy_val))
    ens_ratio = ens / max(1, M)

    has_returns = scenario_data is not None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Scenario Probability Profile (Prior p vs. Posterior x*)",
            "View-Shifted Return Density (Prior vs. Posterior)"
            if has_returns
            else "Cumulative Probability Concentration (Lorenz Curve)",
        ),
        horizontal_spacing=0.10,
    )

    if sort_by_posterior:
        order = np.argsort(posterior)[::-1]
        post_plot = posterior[order]
        prior_plot = prior[order]
        x_label = "Scenario Rank (Sorted by Posterior Weight x*)"
    else:
        post_plot = posterior
        prior_plot = prior
        x_label = "Scenario Index (m = 0 ... M-1)"

    x_idx = np.arange(M)

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=post_plot,
            mode="lines",
            line={"color": palette.violet, "width": 2.2},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.violet, 0.30),
            name="Posterior Probability (x*)",
            hovertemplate="Rank/Idx: %{x}<br>Posterior x*: %{y:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=prior_plot,
            mode="lines",
            line={"color": palette.emerald, "width": 2, "dash": "dash"},
            name=f"Uniform Prior (1/M = {1.0 / max(1, M):.4f})",
            hovertemplate="Prior p: %{y:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Return Density shift if scenario_data is provided, else Lorenz concentration curve
    if scenario_data is not None:
        if isinstance(scenario_data, ScenarioMarketData):
            cum_asset_rets = scenario_data.get_cumulative_returns().astype(np.float64)
            N = cum_asset_rets.shape[1]
            if asset_weights is None:
                w = np.full(N, 1.0 / N)
            elif isinstance(asset_weights, pd.Series):
                w = (
                    asset_weights.reindex(scenario_data.asset_names)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                w = np.asarray(asset_weights, dtype=np.float64)
            scen_returns = cum_asset_rets @ w
        elif isinstance(scenario_data, pd.Series):
            scen_returns = scenario_data.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(scenario_data, dtype=np.float64)
            scen_returns = arr if arr.ndim == 1 else np.mean(arr, axis=1)

        x_prior, y_prior = compute_kde_1d(scen_returns, weights=prior, n_points=250)
        x_post, y_post = compute_kde_1d(scen_returns, weights=posterior, n_points=250)

        prior_mean = float(np.dot(scen_returns, prior))
        post_mean = float(np.dot(scen_returns, posterior))

        fig.add_trace(
            go.Scatter(
                x=x_prior,
                y=y_prior,
                mode="lines",
                line={"color": palette.emerald, "width": 2.2, "dash": "dash"},
                fill="tozeroy",
                fillcolor=hex_to_rgba(palette.emerald, 0.16),
                name=f"Prior Density (Mean={prior_mean:.2%})",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=x_post,
                y=y_post,
                mode="lines",
                line={"color": palette.violet, "width": 2.8},
                fill="tozeroy",
                fillcolor=hex_to_rgba(palette.violet, 0.28),
                name=f"Posterior Density (Mean={post_mean:.2%})",
            ),
            row=1,
            col=2,
        )
        fig.add_vline(
            x=prior_mean,
            line_dash="dash",
            line_color=palette.emerald,
            line_width=1.8,
            row=1,
            col=2,
        )
        fig.add_vline(
            x=post_mean,
            line_dash="solid",
            line_color=palette.violet,
            line_width=2.2,
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Cumulative Scenario Return", tickformat=".1%", row=1, col=2
        )
        fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    else:
        sorted_post = np.sort(posterior)[::-1]
        cum_post = np.cumsum(sorted_post)
        cum_prior = np.cumsum(np.sort(prior)[::-1])
        pct_scenarios = np.arange(1, M + 1) / M

        fig.add_trace(
            go.Scatter(
                x=pct_scenarios,
                y=cum_post,
                mode="lines",
                line={"color": palette.violet, "width": 2.5},
                name="Posterior Concentration",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=pct_scenarios,
                y=cum_prior,
                mode="lines",
                line={"color": palette.emerald, "width": 2, "dash": "dash"},
                name="Uniform Prior Diagonal",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Fraction of Scenarios", tickformat=".0%", row=1, col=2
        )
        fig.update_yaxes(
            title_text="Cumulative Probability Mass", tickformat=".0%", row=1, col=2
        )

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_yaxes(title_text="Scenario Probability Weight", row=1, col=1)

    subtitle = (
        f"KL Divergence: {entropy_result.kl_divergence:.4f} | "
        f"Effective Scenarios (ENS): {ens:.1f} / {M} ({ens_ratio:.1%}) | "
        f"Solver Converged: {entropy_result.success}"
    )
    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=500)
    return fig


def plot_regime_transitions(
    equity_or_scenarios: BacktestResult | pd.Series | ScenarioMarketData,
    gmm_or_probs: GMMRegimeGenerator | pd.DataFrame | np.ndarray | None = None,
    market_data: MarketData | None = None,
    asset_weights: np.ndarray | pd.Series | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Macro Regime Transition Overlay & State Probabilities",
) -> go.Figure:
    """
    Plots the portfolio equity curve against the real-time probability (p_k) of operating
    in each Gaussian Mixture Model (GMM) macro regime (e.g., Bull/Low-Vol vs. Bear/High-Vol).
    Supports both:
    - Historical Mode (`BacktestResult` or `pd.Series` + `GMMRegimeGenerator` or regime probability DataFrame)
    - Synthetic Multiverse Mode (`ScenarioMarketData` with `features['regime_states']`)
    """
    palette = get_theme(theme)
    regime_colors = [
        palette.emerald,
        palette.amber,
        palette.crimson,
        palette.violet,
        palette.cyan,
    ]

    # Case 1: Synthetic Multiverse ScenarioMarketData
    if isinstance(equity_or_scenarios, ScenarioMarketData):
        scen = equity_or_scenarios
        equity_paths, timestamps = extract_equity_paths(
            scen, asset_weights=asset_weights
        )
        x_vals = list(timestamps)
        M, T = equity_paths.shape

        if "regime_states" in scen.features:
            states_sim = scen.features["regime_states"][:, :, 0].astype(int)  # (M, T)
            unique_regimes = sorted(np.unique(states_sim).tolist())
            k_eff = max(len(unique_regimes), int(np.max(states_sim)) + 1)
            probs_arr = np.zeros((T, k_eff), dtype=np.float64)
            for k in range(k_eff):
                probs_arr[:, k] = np.mean(states_sim == k, axis=0)
            labels = (
                gmm_or_probs.regime_labels
                if isinstance(gmm_or_probs, GMMRegimeGenerator)
                and gmm_or_probs.regime_labels
                else [
                    "Bull / Low-Vol"
                    if k == 0
                    else ("Bear / High-Vol" if k == k_eff - 1 else f"Regime {k}")
                    for k in range(k_eff)
                ]
            )
            regime_probs_df = pd.DataFrame(probs_arr, index=timestamps, columns=labels)
        elif isinstance(gmm_or_probs, pd.DataFrame):
            regime_probs_df = gmm_or_probs
        else:
            raise ValueError(
                "ScenarioMarketData does not contain 'regime_states' in features; pass a fitted GMMRegimeGenerator or regime probabilities."
            )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.62, 0.38],
            vertical_spacing=0.07,
            subplot_titles=(
                "Simulated Equity Trajectory (Median & 10th–90th Cone)",
                "Cross-Sectional Regime Probability p_k(t) Across Synthetic Paths",
            ),
        )

        q10, q50, q90 = weighted_quantiles(equity_paths, [0.10, 0.50, 0.90])
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=q90, mode="lines", line={"width": 0}, showlegend=False
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
                fillcolor=hex_to_rgba(palette.cyan, 0.22),
                name="10th–90th Equity Cone",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q50,
                mode="lines",
                line={"color": palette.cyan, "width": 2.8},
                name="Median Portfolio Equity",
            ),
            row=1,
            col=1,
        )

        for idx, col in enumerate(regime_probs_df.columns):
            c_hex = regime_colors[idx % len(regime_colors)]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=regime_probs_df[col],
                    mode="lines",
                    stackgroup="regimes",
                    line={"color": c_hex, "width": 1.2},
                    fillcolor=hex_to_rgba(c_hex, 0.55),
                    name=str(col),
                    hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        apply_plotly_theme(
            fig,
            theme=theme,
            title=title,
            subtitle=f"Dynamic Markov regime proportions across {M:,} simulated paths",
            height=580,
        )
        fig.update_yaxes(title_text="Portfolio Equity", row=1, col=1)
        fig.update_yaxes(
            title_text="Regime Prob p_k(t)",
            tickformat=".0%",
            range=[0, 1],
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Simulation Horizon", row=2, col=1)
        return fig

    # Case 2: Historical BacktestResult or Equity Series
    equity_series = (
        equity_or_scenarios.equity_curve
        if isinstance(equity_or_scenarios, BacktestResult)
        else pd.Series(equity_or_scenarios)
    )

    if isinstance(gmm_or_probs, GMMRegimeGenerator):
        if market_data is not None:
            regime_probs_df = gmm_or_probs.predict_regime_probs(market_data)
        elif gmm_or_probs.historical_regime_probs is not None:
            regime_probs_df = gmm_or_probs.historical_regime_probs
        else:
            raise ValueError(
                "GMMRegimeGenerator must be fitted or `market_data` provided."
            )
    elif isinstance(gmm_or_probs, pd.DataFrame):
        regime_probs_df = gmm_or_probs
    elif isinstance(gmm_or_probs, np.ndarray):
        gmm_or_probs.shape[1] if gmm_or_probs.ndim == 2 else 1
        probs_2d = (
            gmm_or_probs
            if gmm_or_probs.ndim == 2
            else np.column_stack([1.0 - gmm_or_probs, gmm_or_probs])
        )
        regime_probs_df = pd.DataFrame(
            probs_2d,
            index=equity_series.index[: len(probs_2d)],
            columns=[f"Regime {k}" for k in range(probs_2d.shape[1])],
        )
    else:
        raise TypeError(
            "Provide `gmm_or_probs` as a fitted `GMMRegimeGenerator`, DataFrame, or 2D array."
        )

    common_idx = equity_series.index.intersection(regime_probs_df.index)
    if len(common_idx) == 0:
        # Align by position if indices differ in type
        min_len = min(len(equity_series), len(regime_probs_df))
        eq_aligned = equity_series.iloc[-min_len:]
        probs_aligned = regime_probs_df.iloc[-min_len:].set_index(eq_aligned.index)
    else:
        eq_aligned = equity_series.loc[common_idx]
        probs_aligned = regime_probs_df.loc[common_idx]

    x_vals = list(eq_aligned.index)
    dominant_regime = np.argmax(probs_aligned.to_numpy(dtype=np.float64), axis=1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.07,
        subplot_titles=(
            "Portfolio Equity Curve with Dominant Macro Regime Shading",
            "Real-Time Posterior Regime Probabilities p_k(t)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=eq_aligned.to_numpy(dtype=np.float64),
            mode="lines",
            line={"color": palette.text_primary, "width": 2.6},
            name="Portfolio Equity",
            hovertemplate="Equity: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Shade contiguous dominant regime segments on the top equity chart
    if len(x_vals) > 1:
        seg_start = 0
        for t in range(1, len(x_vals) + 1):
            if t == len(x_vals) or dominant_regime[t] != dominant_regime[seg_start]:
                reg_id = int(dominant_regime[seg_start])
                c_hex = regime_colors[reg_id % len(regime_colors)]
                x0 = x_vals[seg_start]
                x1 = x_vals[min(t, len(x_vals) - 1)]
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=hex_to_rgba(c_hex, 0.16),
                    line_width=0,
                    layer="below",
                    row=1,
                    col=1,
                )
                seg_start = t

    # Bottom panel: Stacked area chart of real-time regime probabilities p_k(t)
    for idx, col in enumerate(probs_aligned.columns):
        c_hex = regime_colors[idx % len(regime_colors)]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=probs_aligned[col].to_numpy(dtype=np.float64),
                mode="lines",
                stackgroup="prob_stack",
                line={"color": c_hex, "width": 1.2},
                fillcolor=hex_to_rgba(c_hex, 0.55),
                name=str(col),
                hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    regime_shares = probs_aligned.mean(axis=0)
    share_str = " | ".join(f"{col}: {val:.1%}" for col, val in regime_shares.items())

    apply_plotly_theme(
        fig,
        theme=theme,
        title=title,
        subtitle=f"Average Regime Occupancy — {share_str}",
        height=580,
    )
    fig.update_yaxes(title_text="Cumulative Equity", row=1, col=1)
    fig.update_yaxes(
        title_text="Probability p_k(t)", tickformat=".0%", range=[0, 1], row=2, col=1
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig
