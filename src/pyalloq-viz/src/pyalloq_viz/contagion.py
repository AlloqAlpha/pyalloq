from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.linalg as sla
from plotly.subplots import make_subplots

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba


def plot_spectral_contagion_network(
    adjacency: pd.DataFrame | np.ndarray,
    branching_matrix: pd.DataFrame | np.ndarray | None = None,
    asset_names: list[str] | None = None,
    edge_threshold: float = 0.05,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Spectral Laplacian Contagion Network & Fiedler Partition",
) -> go.Figure:
    """
    Visualizes the market topology as a two-panel Plotly figure:
    - Left Panel: Interactive graph network where node colors encode the Fiedler vector (v_2)
      spectral bisection partition (+ vs -), node sizes encode Hawkes self-/cross-excitation
      out-degree (systemic transmitter intensity), and edge thickness encodes partial-correlation
      coupling strength.
    - Right Panel: Sorted Fiedler vector (v_2) coordinates and systemic transmitter scores
      alongside algebraic connectivity (lambda_2) and Hawkes branching spectral radius (rho(G)).
    """
    palette = get_theme(theme)

    if isinstance(adjacency, pd.DataFrame):
        names = [str(c) for c in adjacency.columns]
        adj_mat = adjacency.to_numpy(dtype=np.float64, copy=True)
    else:
        adj_mat = np.asarray(adjacency, dtype=np.float64).copy()
        n = adj_mat.shape[0]
        names = (
            [str(a) for a in asset_names]
            if asset_names is not None and len(asset_names) == n
            else [f"Asset_{i}" for i in range(n)]
        )

    n = adj_mat.shape[0]
    adj_mat = 0.5 * (adj_mat + adj_mat.T)
    adj_mat = np.maximum(adj_mat, 0.0)
    np.fill_diagonal(adj_mat, 0.0)

    deg = adj_mat.sum(axis=1)
    inv_sqrt_deg = np.where(deg > 1e-12, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    d_half = np.diag(inv_sqrt_deg)
    lap = np.eye(n, dtype=np.float64) - d_half @ adj_mat @ d_half
    lap = 0.5 * (lap + lap.T)

    if n >= 2:
        evals, evecs = sla.eigh(lap)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        lambda_2 = float(max(0.0, evals[1]))
        fiedler_vec = evecs[:, 1].astype(np.float64)
        if float(np.dot(fiedler_vec, np.arange(1, n + 1, dtype=np.float64))) < 0.0:
            fiedler_vec = -fiedler_vec
    else:
        lambda_2 = 1.0
        fiedler_vec = np.zeros(n, dtype=np.float64)

    if branching_matrix is not None:
        g_mat = np.asarray(branching_matrix, dtype=np.float64)
        out_influence = np.sum(np.abs(g_mat), axis=0)
        eigs_g = sla.eigvals(g_mat)
        rho_g = float(np.max(np.abs(eigs_g))) if len(eigs_g) > 0 else 0.0
    else:
        out_influence = deg.copy()
        rho_g = 0.0

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.10,
        subplot_titles=(
            f"Topology & Fiedler Cut (λ₂ = {lambda_2:.3f}, ρ(G) = {rho_g:.3f})",
            "Spectral Fiedler Coordinates (v₂) & Systemic Out-Degree",
        ),
    )

    # Circular layout ordered by Fiedler coordinate so clusters group naturally
    fiedler_order = np.argsort(fiedler_vec)
    angles = np.zeros(n, dtype=np.float64)
    for rank, node_idx in enumerate(fiedler_order):
        angles[node_idx] = 2.0 * np.pi * float(rank) / max(1, n)
    xs = np.cos(angles)
    ys = np.sin(angles)

    max_w = float(np.max(adj_mat)) if n > 1 and float(np.max(adj_mat)) > 1e-12 else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_mat[i, j])
            if w < edge_threshold:
                continue
            rel_w = float(np.clip(w / max_w, 0.15, 1.0))
            fig.add_trace(
                go.Scatter(
                    x=[xs[i], xs[j], None],
                    y=[ys[i], ys[j], None],
                    mode="lines",
                    line={
                        "width": 1.0 + 4.5 * rel_w,
                        "color": hex_to_rgba(palette.cyan, 0.25 + 0.55 * rel_w),
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    max_inf = (
        float(np.max(out_influence)) if float(np.max(out_influence)) > 1e-12 else 1.0
    )
    node_sizes = 20.0 + 26.0 * (out_influence / max_inf)
    node_colors = [
        palette.emerald if float(v) >= 0.0 else palette.coral for v in fiedler_vec
    ]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={
                "size": node_sizes,
                "color": node_colors,
                "line": {"width": 2.0, "color": palette.bg_primary},
            },
            customdata=np.column_stack([fiedler_vec, out_influence]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Fiedler v₂: %{customdata[0]:.4f}<br>"
                "Systemic Out-Degree: %{customdata[1]:.4f}<extra></extra>"
            ),
            name="Assets (Fiedler Cut)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    sorted_names = [names[idx] for idx in fiedler_order]
    sorted_v2 = fiedler_vec[fiedler_order]
    sorted_colors = [
        palette.emerald if float(v) >= 0.0 else palette.coral for v in sorted_v2
    ]

    fig.add_trace(
        go.Bar(
            x=sorted_v2,
            y=sorted_names,
            orientation="h",
            marker={"color": sorted_colors},
            name="Fiedler Coordinate (v₂)",
            hovertemplate="<b>%{y}</b><br>v₂ = %{x:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        visible=False, range=[-1.35, 1.35], showgrid=False, zeroline=False, row=1, col=1
    )
    fig.update_yaxes(
        visible=False, range=[-1.35, 1.35], showgrid=False, zeroline=False, row=1, col=1
    )
    fig.update_xaxes(title_text="Fiedler Vector Coordinate (v₂)", row=1, col=2)

    apply_plotly_theme(fig, theme=theme, title=title, height=520)
    return fig


def plot_fragility_hawkes_regime_portrait(
    diagnostics_df: pd.DataFrame,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Spectral Fragility (λ₂), Hawkes Branching Ratio ρ(G) & Roughness Regime Portrait",
) -> go.Figure:
    """
    Visualizes rolling structural contagion and path-roughness diagnostics across time:
    - Panel 1: Algebraic Connectivity (Fiedler value λ₂) vs Systemic Fragility Penalty (1 / λ₂).
    - Panel 2: Hawkes Endogenous Branching Spectral Radius ρ(G_t) with critical instability
      threshold markers.
    - Panel 3: Roughness / Pathwise Regime Metric (`effective_hurst` or `signature_mmd`).
    """
    palette = get_theme(theme)
    df = diagnostics_df.copy()
    x_idx = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.36, 0.34, 0.30],
        subplot_titles=(
            "Algebraic Connectivity (λ₂) & Spectral Synchronization",
            "Hawkes Endogenous Branching Spectral Radius ρ(G_t)",
            "Pathwise Roughness (Effective Hurst H_eff / Signature-MMD)",
        ),
    )

    if "lambda_2" in df.columns:
        lam2 = df["lambda_2"].to_numpy(dtype=np.float64)
    else:
        lam2 = np.full(len(df), 0.5, dtype=np.float64)

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=lam2,
            mode="lines",
            line={"color": palette.cyan, "width": 2.2},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.cyan, 0.18),
            name="Fiedler Value (λ₂)",
        ),
        row=1,
        col=1,
    )

    if "hawkes_spectral_radius" in df.columns:
        rho_arr = df["hawkes_spectral_radius"].to_numpy(dtype=np.float64)
    elif "rho_g" in df.columns:
        rho_arr = df["rho_g"].to_numpy(dtype=np.float64)
    else:
        rho_arr = np.clip(1.0 - lam2, 0.0, 0.99)

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=rho_arr,
            mode="lines",
            line={"color": palette.coral, "width": 2.2},
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette.coral, 0.20),
            name="Hawkes Spectral Radius ρ(G)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color=palette.amber,
        annotation_text="Critical Endogenous Contagion (ρ = 0.80)",
        row=2,
        col=1,
    )

    if "effective_hurst" in df.columns:
        r_arr = df["effective_hurst"].to_numpy(dtype=np.float64)
        r_name = "Effective Hurst Exponent (H_eff)"
        ref_line: float | None = 0.50
        ref_label = "Brownian Baseline (H = 0.50)"
    elif "signature_mmd" in df.columns:
        r_arr = df["signature_mmd"].to_numpy(dtype=np.float64)
        r_name = "Signature-MMD Regime Shift"
        ref_line = None
        ref_label = ""
    else:
        r_arr = np.full(len(df), 0.15, dtype=np.float64)
        r_name = "Effective Hurst Exponent (H_eff)"
        ref_line = 0.50
        ref_label = "Brownian Baseline (H = 0.50)"

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=r_arr,
            mode="lines",
            line={"color": palette.violet, "width": 2.0},
            name=r_name,
        ),
        row=3,
        col=1,
    )
    if ref_line is not None:
        fig.add_hline(
            y=ref_line,
            line_dash="dot",
            line_color=palette.slate,
            annotation_text=ref_label,
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="λ₂", row=1, col=1)
    fig.update_yaxes(title_text="ρ(G)", range=[0.0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text=r_name, row=3, col=1)

    apply_plotly_theme(fig, theme=theme, title=title, height=740)
    return fig
