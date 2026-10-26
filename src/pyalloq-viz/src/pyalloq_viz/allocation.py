from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from plotly.subplots import make_subplots
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult, OptimizationResult
from pyalloq_core.utils import cov_to_corr
from scipy.optimize import minimize

from pyalloq_viz.themes import ThemeName, apply_plotly_theme, get_theme, hex_to_rgba
from pyalloq_viz.utils import compute_risk_contributions, reconstruct_drifted_weights


def _solve_parametric_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the classical long-only Markowitz Efficient Frontier (vols, returns, weights).
    """
    N = len(mu)
    bounds = [(0.0, 1.0) for _ in range(N)]
    w0 = np.full(N, 1.0 / N)
    cons_sum: dict[str, Any] = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    # Find Min Volatility portfolio
    res_min = minimize(  # type: ignore[call-overload]
        lambda w: float(w.T @ cov @ w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[cons_sum],
    )
    w_min = res_min.x if res_min.success else w0
    min_ret = float(w_min @ mu)
    max_ret = float(np.max(mu))

    target_rets = np.linspace(min_ret, max_ret * 0.995, n_points)
    frontier_vols: list[float] = []
    frontier_rets: list[float] = []
    frontier_weights: list[np.ndarray] = []

    for r_target in target_rets:
        cons: list[dict[str, Any]] = [
            cons_sum,
            {"type": "eq", "fun": lambda w, rt=r_target: float(w @ mu) - float(rt)},
        ]
        res = minimize(  # type: ignore[call-overload]
            lambda w: float(w.T @ cov @ w),
            w_min,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        if res.success or res.x is not None:
            w_opt = np.clip(res.x, 0.0, 1.0)
            w_opt /= max(float(np.sum(w_opt)), 1e-12)
            vol = float(np.sqrt(max(float(w_opt.T @ cov @ w_opt), 0.0)))
            ret = float(w_opt @ mu)
            frontier_vols.append(vol)
            frontier_rets.append(ret)
            frontier_weights.append(w_opt)

    return (
        np.asarray(frontier_vols),
        np.asarray(frontier_rets),
        np.asarray(frontier_weights),
    )


def plot_multi_universe_frontier(
    expected_returns: pd.Series | np.ndarray | None = None,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
    scenario_data: ScenarioMarketData | None = None,
    allocator_or_pipeline: BaseAllocator | StrategyPipeline | None = None,
    optimal_portfolios: OptimizationResult
    | Sequence[OptimizationResult]
    | pd.DataFrame
    | None = None,
    n_random_portfolios: int = 400,
    max_scenario_solves: int = 80,
    risk_free_rate: float = 0.0,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Multi-Universe Efficient Frontier & Optimal Allocation Cloud",
) -> go.Figure:
    """
    Renders a risk/return scatter plot combining:
    1. The parametric Markowitz Efficient Frontier curve (and individual asset coordinates).
    2. A Random Dirichlet Portfolio Cloud colored by Sharpe ratio.
    3. A Multi-Universe Optimal Allocation Cloud generated across synthetic realities
       (`ScenarioMarketData` + `allocator_or_pipeline` or precomputed `optimal_portfolios`).
    """
    palette = get_theme(theme)
    ann_factor = 252.0

    # Infer expected_returns and cov_matrix from scenario_data if not directly provided
    asset_names: list[str] = []
    if cov_matrix is not None:
        if isinstance(cov_matrix, pd.DataFrame):
            asset_names = [str(c) for c in cov_matrix.columns]
            Sigma = cov_matrix.to_numpy(dtype=np.float64)
        else:
            Sigma = np.asarray(cov_matrix, dtype=np.float64)
            asset_names = [f"Asset_{i}" for i in range(Sigma.shape[0])]
    elif scenario_data is not None:
        asset_names = list(scenario_data.asset_names)
        mean_path_rets = np.mean(
            scenario_data.returns.astype(np.float64), axis=0
        )  # (T, N)
        Sigma = np.cov(mean_path_rets, rowvar=False) * ann_factor
    else:
        raise ValueError(
            "Provide `cov_matrix` or `scenario_data` to plot the Efficient Frontier."
        )

    N = Sigma.shape[0]
    if expected_returns is not None:
        if isinstance(expected_returns, pd.Series):
            mu = (
                expected_returns.reindex(asset_names)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            mu = np.asarray(expected_returns, dtype=np.float64)
    elif scenario_data is not None:
        mu = np.mean(scenario_data.returns.astype(np.float64), axis=(0, 1)) * ann_factor
    else:
        mu = np.sqrt(np.diag(Sigma)) * 0.6

    fig = go.Figure()

    # 1. Random Dirichlet Portfolio Cloud
    if n_random_portfolios > 0 and N >= 2:
        rng = np.random.default_rng(42)
        rand_w = rng.dirichlet(np.ones(N), size=n_random_portfolios)
        rand_rets = rand_w @ mu
        rand_vols = np.sqrt(np.einsum("mi,ij,mj->m", rand_w, Sigma, rand_w))
        rand_sharpes = (rand_rets - risk_free_rate) / np.maximum(rand_vols, 1e-12)

        fig.add_trace(
            go.Scatter(
                x=rand_vols,
                y=rand_rets,
                mode="markers",
                marker={
                    "size": 6,
                    "color": rand_sharpes,
                    "colorscale": "Tealgrn",
                    "opacity": 0.45,
                    "colorbar": {"title": "Sharpe", "thickness": 12, "len": 0.65},
                },
                name="Random Feasible Portfolios",
                hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    # 2. Parametric Markowitz Efficient Frontier Curve
    if N >= 2:
        f_vols, f_rets, _ = _solve_parametric_frontier(mu, Sigma, n_points=35)
        fig.add_trace(
            go.Scatter(
                x=f_vols,
                y=f_rets,
                mode="lines",
                line={"color": palette.emerald, "width": 3},
                name="Parametric Efficient Frontier",
                hovertemplate="Frontier Vol: %{x:.2%}<br>Frontier Return: %{y:.2%}<extra></extra>",
            )
        )

    # 3. Individual Assets
    asset_vols = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    fig.add_trace(
        go.Scatter(
            x=asset_vols,
            y=mu,
            mode="markers+text",
            text=asset_names,
            textposition="top center",
            textfont={"size": 10, "color": palette.text_secondary},
            marker={
                "size": 10,
                "symbol": "diamond",
                "color": palette.cyan,
                "line": {"width": 1.5, "color": palette.bg_primary},
            },
            name="Individual Assets",
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Expected Return: %{y:.2%}<extra></extra>",
        )
    )

    # 4. Multi-Universe Optimal Allocation Cloud across Synthetic Realities
    sim_vols: list[float] = []
    sim_rets: list[float] = []

    if scenario_data is not None:
        M_paths = min(scenario_data.n_paths, max_scenario_solves)
        for m in range(M_paths):
            path_rets = scenario_data.returns[m].astype(np.float64)  # (T, N)
            mu_m = np.mean(path_rets, axis=0) * ann_factor
            cov_m = np.cov(path_rets, rowvar=False) * ann_factor
            if cov_m.ndim == 0:
                cov_m = np.array([[float(cov_m)]])

            w_m: np.ndarray | None = None
            if isinstance(allocator_or_pipeline, StrategyPipeline):
                path_md = scenario_data.get_path(m)
                w_ser = allocator_or_pipeline.generate_weights(path_md)
                w_m = w_ser.reindex(asset_names).fillna(0.0).to_numpy(dtype=np.float64)
            elif isinstance(allocator_or_pipeline, BaseAllocator):
                path_md = scenario_data.get_path(m)
                cov_df = pd.DataFrame(cov_m, index=asset_names, columns=asset_names)
                mu_ser = pd.Series(mu_m, index=asset_names)
                try:
                    opt_res = allocator_or_pipeline.allocate(
                        data=path_md, cov_matrix=cov_df, expected_returns=mu_ser
                    )
                    w_m = (
                        opt_res.weights.reindex(asset_names)
                        .fillna(0.0)
                        .to_numpy(dtype=np.float64)
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    w_m = None
            else:
                # Default: compute min-variance / tangency allocation on scenario m
                inv_diag = 1.0 / np.maximum(np.diag(cov_m), 1e-8)
                w_m = inv_diag / np.sum(inv_diag)

            if w_m is not None and np.sum(w_m) > 0:
                w_m = w_m / np.sum(w_m)
                # Plot scenario-optimal portfolio evaluated under scenario m's realized dynamics
                sim_vols.append(float(np.sqrt(max(float(w_m.T @ cov_m @ w_m), 0.0))))
                sim_rets.append(float(w_m @ mu_m))

    if len(sim_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=sim_vols,
                y=sim_rets,
                mode="markers",
                marker={
                    "size": 9,
                    "color": palette.violet,
                    "opacity": 0.82,
                    "symbol": "circle",
                    "line": {"width": 1, "color": palette.text_primary},
                },
                name=f"Multi-Universe Optimal Cloud ({len(sim_vols)} Realities)",
                hovertemplate="Scenario Optimal Vol: %{x:.2%}<br>Scenario Optimal Return: %{y:.2%}<extra></extra>",
            )
        )

    # 5. Highlighted Optimal Portfolio(s)
    if optimal_portfolios is not None:
        opt_list: list[tuple[str, np.ndarray]] = []
        if isinstance(optimal_portfolios, OptimizationResult):
            w_arr = (
                optimal_portfolios.weights.reindex(asset_names)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            opt_list.append(
                (str(getattr(optimal_portfolios, "status", "Optimal Portfolio")), w_arr)
            )
        elif isinstance(optimal_portfolios, pd.DataFrame):
            for idx_name, row in optimal_portfolios.iterrows():
                opt_list.append(
                    (
                        str(idx_name),
                        row.reindex(asset_names).fillna(0.0).to_numpy(dtype=np.float64),
                    )
                )
        else:
            for i, opt in enumerate(optimal_portfolios):
                w_arr = (
                    opt.weights.reindex(asset_names)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                opt_list.append((f"{opt.status} ({i+1})", w_arr))

        for label, w_arr in opt_list:
            p_vol = float(np.sqrt(max(float(w_arr.T @ Sigma @ w_arr), 0.0)))
            p_ret = float(w_arr @ mu)
            fig.add_trace(
                go.Scatter(
                    x=[p_vol],
                    y=[p_ret],
                    mode="markers",
                    marker={
                        "size": 15,
                        "symbol": "star",
                        "color": palette.amber,
                        "line": {"width": 1.5, "color": palette.bg_primary},
                    },
                    name=label,
                    hovertemplate=f"<b>{label}</b><br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
                )
            )

    subtitle = f"Universe of {N} assets"
    if len(sim_vols) > 0:
        subtitle += f" | {len(sim_vols)} synthetic optimal allocations rendered"

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=540)
    fig.update_xaxes(title_text="Annualized Volatility (σ)", tickformat=".1%")
    fig.update_yaxes(title_text="Annualized Expected Return (μ)", tickformat=".1%")
    return fig


def plot_weight_drift_turnover(
    backtest_result: BacktestResult | pd.DataFrame,
    market_data: MarketData | pd.DataFrame | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Asset Weight Drift & Rebalance Turnover Dynamics",
) -> go.Figure:
    """
    Plots a two-panel view of portfolio allocation mechanics over time:
    - Top Panel: Stacked area chart of asset weights drifting naturally due to daily
      market movements and snapping back during rebalancing events.
    - Bottom Panel: Bar chart of two-way portfolio turnover at each rebalance event
      alongside cumulative turnover.
    """
    palette = get_theme(theme)
    drifted_df, turnover_series = reconstruct_drifted_weights(
        backtest_result, market_data=market_data
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.66, 0.34],
        vertical_spacing=0.07,
        subplot_titles=(
            "Drifted Asset Weights & Rebalance Snap-Backs",
            "Rebalance Two-Way Turnover (∑ |Δw_i|)",
        ),
    )

    x_vals = list(drifted_df.index)
    for idx, asset in enumerate(drifted_df.columns):
        c_hex = palette.categorical[idx % len(palette.categorical)]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=drifted_df[asset].to_numpy(dtype=np.float64),
                mode="lines",
                stackgroup="weights",
                line={"color": c_hex, "width": 1.0},
                fillcolor=hex_to_rgba(c_hex, 0.72),
                name=str(asset),
                hovertemplate=f"<b>{asset}</b>: %{{y:.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Filter non-zero rebalance events for cleaner bar chart rendering
    reb_events = turnover_series[turnover_series > 1e-5]
    fig.add_trace(
        go.Bar(
            x=list(reb_events.index),
            y=reb_events.to_numpy(dtype=np.float64),
            marker={
                "color": palette.amber,
                "line": {"color": palette.amber, "width": 1},
            },
            name="Rebalance Turnover",
            hovertemplate="Date: %{x}<br>Turnover: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    total_turnover = (
        float(turnover_series.iloc[1:].sum()) if len(turnover_series) > 1 else 0.0
    )
    n_rebalances = (
        int((turnover_series.iloc[1:] > 1e-5).sum()) if len(turnover_series) > 1 else 0
    )

    subtitle = (
        f"Rebalance Events: {n_rebalances} | "
        f"Cumulative Post-Init Turnover: {total_turnover:.1%} | "
        f"Avg Turnover / Rebalance: {(total_turnover / max(1, n_rebalances)):.2%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=580)
    fig.update_yaxes(
        title_text="Portfolio Weight", tickformat=".0%", range=[0.0, 1.0], row=1, col=1
    )
    fig.update_yaxes(title_text="Turnover (%)", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def plot_hierarchical_composition(
    weights_or_result: OptimizationResult | pd.Series,
    target_or_benchmark_weights: OptimizationResult | pd.Series | None = None,
    hierarchy: Mapping[str, str | Sequence[str]] | pd.DataFrame | None = None,
    market_data: MarketData | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Hierarchical Portfolio Composition & Target vs. Current Allocation",
) -> go.Figure:
    """
    Left Panel: Multi-level Treemap displaying hierarchical portfolio structure
      (e.g., Portfolio -> Sector/Cluster -> Asset), resolving hierarchy from explicit
      `hierarchy` mapping, `MarketData.cross_sectional`, `OptimizationResult.metadata['clusters']`,
      or flat fallback.
    Right Panel: Grouped horizontal comparison bar chart of Current/Optimized vs. Target/Benchmark weights.
    """
    palette = get_theme(theme)

    if isinstance(weights_or_result, OptimizationResult):
        w_series = weights_or_result.weights.copy()
        meta_clusters = weights_or_result.metadata.get("clusters")
    else:
        w_series = pd.Series(weights_or_result).copy()
        meta_clusters = None

    w_series = w_series[w_series > 1e-6]
    assets = [str(a) for a in w_series.index]

    # Resolve parent group for each asset
    parent_map: dict[str, str] = {}
    if isinstance(hierarchy, pd.DataFrame):
        col = hierarchy.columns[0]
        for a in assets:
            if a in hierarchy.index:
                parent_map[a] = str(hierarchy.loc[a, col])
    elif isinstance(hierarchy, Mapping):
        for k, v in hierarchy.items():
            if isinstance(v, (list, tuple)) and not isinstance(v, str):
                parent_map[str(k)] = " / ".join(str(x) for x in v)
            else:
                parent_map[str(k)] = str(v)
    elif (
        market_data is not None
        and market_data.cross_sectional is not None
        and not market_data.cross_sectional.empty
    ):
        cs = market_data.cross_sectional
        sector_col = next(
            (
                c
                for c in cs.columns
                if "sector" in str(c).lower()
                or "group" in str(c).lower()
                or "class" in str(c).lower()
            ),
            cs.columns[0],
        )
        for a in assets:
            if a in cs.index:
                parent_map[a] = str(cs.loc[a, sector_col])
    elif isinstance(meta_clusters, Mapping):
        for a in assets:
            if a in meta_clusters:
                parent_map[a] = str(meta_clusters[a])

    # Build Treemap nodes (Root -> Group -> Asset)
    root_label = "Portfolio"
    ids: list[str] = [root_label]
    labels: list[str] = [root_label]
    parents: list[str] = [""]
    values: list[float] = [float(w_series.sum())]

    if parent_map:
        group_totals: dict[str, float] = {}
        for k_asset, val in w_series.items():
            g = parent_map.get(str(k_asset), "Other")
            group_totals[g] = group_totals.get(g, 0.0) + float(val)

        for g, g_val in group_totals.items():
            g_id = f"{root_label}/{g}"
            ids.append(g_id)
            labels.append(g)
            parents.append(root_label)
            values.append(g_val)

        for k_asset, val in w_series.items():
            a_str = str(k_asset)
            g = parent_map.get(a_str, "Other")
            g_id = f"{root_label}/{g}"
            ids.append(f"{g_id}/{a_str}")
            labels.append(a_str)
            parents.append(g_id)
            values.append(float(val))
    else:
        for k_asset, val in w_series.items():
            a_str = str(k_asset)
            ids.append(f"{root_label}/{a_str}")
            labels.append(a_str)
            parents.append(root_label)
            values.append(float(val))

    # Resolve target / benchmark comparison weights
    if target_or_benchmark_weights is not None:
        if isinstance(target_or_benchmark_weights, OptimizationResult):
            bench_w = target_or_benchmark_weights.weights.reindex(
                w_series.index
            ).fillna(0.0)
        else:
            bench_w = (
                pd.Series(target_or_benchmark_weights)
                .reindex(w_series.index)
                .fillna(0.0)
            )
        bench_label = "Target / Benchmark Weight"
    else:
        bench_w = pd.Series(1.0 / max(1, len(w_series)), index=w_series.index)
        bench_label = "Equal-Weight Benchmark (1/N)"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.54, 0.46],
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=(
            "Hierarchical Composition Treemap",
            "Current vs. Target / Benchmark Allocation",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo="label+percent root",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>",
            marker={
                "colorscale": "Tealgrn",
                "line": {"width": 1.5, "color": palette.bg_primary},
            },
        ),
        row=1,
        col=1,
    )

    sorted_idx = w_series.sort_values(ascending=True).index
    fig.add_trace(
        go.Bar(
            y=[str(a) for a in sorted_idx],
            x=w_series.loc[sorted_idx].to_numpy(dtype=np.float64),
            orientation="h",
            marker={"color": palette.emerald},
            name="Optimized Weight",
            hovertemplate="<b>%{y}</b><br>Optimized: %{x:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=[str(a) for a in sorted_idx],
            x=bench_w.loc[sorted_idx].to_numpy(dtype=np.float64),
            orientation="h",
            marker={"color": palette.amber},
            name=bench_label,
            hovertemplate="<b>%{y}</b><br>Target/Bench: %{x:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    eff_n = float(1.0 / np.sum(np.square(w_series.to_numpy(dtype=np.float64))))
    subtitle = (
        f"Active Assets: {len(w_series)} | "
        f"Effective Number of Bets (1 / ∑w²): {eff_n:.2f} | "
        f"Max Single Weight: {float(w_series.max()):.1%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=520)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Allocation Weight (%)", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Asset", row=1, col=2)
    return fig


def plot_risk_contributions(
    weights_or_result: OptimizationResult | pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    target_risk_budgets: pd.Series | np.ndarray | None = None,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Capital Allocation vs. Component Risk Contributions",
) -> go.Figure:
    """
    Compares Capital Weights (w_i), Percentage Component Risk Contributions
    (PCRC_i = w_i * (Sigma w)_i / (w^T Sigma w)), and Target Risk Budgets (b_i).
    Crucial for validating Risk Parity, Risk Budgeting, HRP, and HERC strategies.
    """
    palette = get_theme(theme)

    w: pd.Series | np.ndarray
    if isinstance(weights_or_result, OptimizationResult):
        w = weights_or_result.weights
        if (
            target_risk_budgets is None
            and "target_budgets" in weights_or_result.metadata
        ):
            target_risk_budgets = np.asarray(
                weights_or_result.metadata["target_budgets"], dtype=np.float64
            )
    else:
        w = weights_or_result

    rc_df = compute_risk_contributions(w, cov_matrix)
    assets = [str(a) for a in rc_df.index]
    N = len(assets)

    if target_risk_budgets is not None:
        if isinstance(target_risk_budgets, pd.Series):
            budgets = (
                target_risk_budgets.reindex(rc_df.index)
                .fillna(1.0 / N)
                .to_numpy(dtype=np.float64)
            )
        else:
            budgets = np.asarray(target_risk_budgets, dtype=np.float64)
    else:
        budgets = np.full(N, 1.0 / max(1, N), dtype=np.float64)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=assets,
            y=rc_df["weight"].to_numpy(dtype=np.float64),
            name="Capital Weight (w_i)",
            marker={"color": palette.cyan},
            hovertemplate="<b>%{x}</b><br>Capital Weight: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=assets,
            y=rc_df["pcrc"].to_numpy(dtype=np.float64),
            name="Risk Contribution (PCRC_i)",
            marker={"color": palette.coral},
            hovertemplate="<b>%{x}</b><br>Risk Contribution: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=assets,
            y=budgets,
            mode="lines+markers",
            line={"color": palette.amber, "width": 2.5, "dash": "dash"},
            marker={"size": 9, "symbol": "diamond", "color": palette.amber},
            name="Target Risk Budget (b_i)",
            hovertemplate="<b>%{x}</b><br>Target Budget: %{y:.2%}<extra></extra>",
        )
    )

    pcrc_vals = np.clip(rc_df["pcrc"].to_numpy(dtype=np.float64), 1e-12, None)
    eff_risk_bets = float(1.0 / np.sum(np.square(pcrc_vals)))
    port_vol = float(np.sum(rc_df["crc"]))

    subtitle = (
        f"Portfolio Volatility: {port_vol:.2%} | "
        f"Effective Number of Risk Bets (ENCB): {eff_risk_bets:.2f} / {N} | "
        f"Max Risk Concentration: {float(np.max(pcrc_vals)):.1%}"
    )

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=480)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Asset")
    fig.update_yaxes(title_text="Share of Capital / Risk (%)", tickformat=".0%")
    return fig


def plot_correlation_and_periodic_returns(
    market_data_or_cov: MarketData | pd.DataFrame | np.ndarray,
    portfolio_returns: BacktestResult | pd.Series | None = None,
    cluster_order: bool = True,
    theme: ThemeName | str = "alloq_dark",
    title: str = "Asset Correlation Matrix & Monthly Periodic Return Heatmap",
) -> go.Figure:
    """
    Two-panel diagnostic figure:
    - Left Panel: Asset Return Correlation Matrix Heatmap (optionally quasi-diagonalized
      via hierarchical clustering so correlated asset blocks stand out visually).
    - Right Panel: Monthly Periodic Return Table Heatmap (Jan-Dec across x-axis,
      Years on y-axis) to visualize seasonal performance and annual compounding.
    """
    get_theme(theme)

    # Extract correlation matrix
    if isinstance(market_data_or_cov, MarketData):
        rets_df = market_data_or_cov.get_returns()
        corr_df = rets_df.corr()
        if portfolio_returns is None:
            portfolio_returns = rets_df.mean(axis=1)
    elif isinstance(market_data_or_cov, pd.DataFrame):
        # Check if it's already a square covariance/correlation matrix or a time series of prices/returns
        if market_data_or_cov.shape[0] == market_data_or_cov.shape[1] and list(
            market_data_or_cov.index
        ) == list(market_data_or_cov.columns):
            corr_np = cov_to_corr(market_data_or_cov.to_numpy(dtype=np.float64))
            corr_df = pd.DataFrame(
                corr_np,
                index=market_data_or_cov.index,
                columns=market_data_or_cov.columns,
            )
        else:
            corr_df = market_data_or_cov.pct_change().dropna().corr()
    else:
        corr_np = cov_to_corr(np.asarray(market_data_or_cov, dtype=np.float64))
        names = [f"Asset_{i}" for i in range(corr_np.shape[0])]
        corr_df = pd.DataFrame(corr_np, index=names, columns=names)

    if cluster_order and corr_df.shape[0] > 2:
        try:
            dist = np.sqrt(
                np.clip(0.5 * (1.0 - corr_df.to_numpy(dtype=np.float64)), 0.0, 1.0)
            )
            np.fill_diagonal(dist, 0.0)
            link = sch.linkage(ssd.squareform(dist, checks=False), method="single")
            leaves = sch.leaves_list(link)
            corr_df = corr_df.iloc[leaves, leaves]
        except (ValueError, np.linalg.LinAlgError):
            corr_df = corr_df.copy()

    # Extract portfolio returns for the monthly heatmap
    ret_series: pd.Series | None = None
    if isinstance(portfolio_returns, BacktestResult):
        ret_series = portfolio_returns.returns.dropna()
    elif isinstance(portfolio_returns, pd.Series):
        ret_series = portfolio_returns.dropna()

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.54],
        subplot_titles=(
            "Quasi-Diagonalized Asset Correlation Heatmap"
            if cluster_order
            else "Asset Correlation Matrix Heatmap",
            "Monthly Periodic Returns (%) — Seasonal Performance",
        ),
        horizontal_spacing=0.10,
    )

    corr_vals = corr_df.to_numpy(dtype=np.float64)
    corr_labels = [str(c) for c in corr_df.columns]

    fig.add_trace(
        go.Heatmap(
            z=corr_vals,
            x=corr_labels,
            y=corr_labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=np.round(corr_vals, 2),
            texttemplate="%{text:.2f}",
            showscale=False,
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Compute Monthly Return Grid (Years x Jan..Dec)
    if ret_series is not None and len(ret_series) > 0:
        if not isinstance(ret_series.index, pd.DatetimeIndex):
            # Create synthetic daily business dates starting 2022-01-03 if RangeIndex
            ret_series = ret_series.copy()
            ret_series.index = pd.bdate_range("2022-01-03", periods=len(ret_series))

        monthly = (1.0 + ret_series).resample("ME").prod() - 1.0
        m_dt_idx = pd.DatetimeIndex(monthly.index)
        df_m = pd.DataFrame(
            {
                "Year": m_dt_idx.year,
                "Month": m_dt_idx.month,
                "Return": monthly.to_numpy(dtype=np.float64),
            }
        )
        pivot = df_m.pivot(index="Year", columns="Month", values="Return")
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[list(range(1, 13))]
        pivot.columns = [calendar.month_abbr[m] for m in range(1, 13)]

        # Add Annual Compounded column
        annual = (1.0 + ret_series).resample("YE").prod() - 1.0
        a_dt_idx = pd.DatetimeIndex(annual.index)
        pivot["Year Total"] = [
            float(annual[a_dt_idx.year == yr].iloc[0])
            if any(a_dt_idx.year == yr)
            else np.nan
            for yr in pivot.index
        ]

        z_mat = pivot.to_numpy(dtype=np.float64) * 100.0
        text_mat = np.where(np.isfinite(z_mat), np.char.mod("%.1f%%", z_mat), "")
        y_years = [str(y) for y in pivot.index]
        x_months = list(pivot.columns)

        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=x_months,
                y=y_years,
                colorscale="RdYlGn",
                zmid=0.0,
                text=text_mat,
                texttemplate="%{text}",
                showscale=True,
                colorbar={"title": "Ret (%)", "thickness": 12, "len": 0.75},
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    avg_off_diag_corr = (
        float(
            (np.sum(corr_vals) - np.trace(corr_vals))
            / max(1, corr_vals.size - len(corr_labels))
        )
        if len(corr_labels) > 1
        else 0.0
    )
    subtitle = f"Assets: {len(corr_labels)} | Mean Pairwise Correlation: {avg_off_diag_corr:.2f}"

    apply_plotly_theme(fig, theme=theme, title=title, subtitle=subtitle, height=520)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    return fig
