from __future__ import annotations

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult
from scipy.stats import gaussian_kde


def weighted_quantiles(
    values: np.ndarray,
    quantiles: list[float] | np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes probability-weighted quantiles along 1D or 2D arrays.
    If `values` is 2D of shape (M, T), computes quantiles across axis=0 for each step t=0..T-1,
    returning an array of shape (len(quantiles), T).
    """
    q_arr = np.asarray(quantiles, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)

    if sample_weight is None:
        return np.quantile(vals, q_arr, axis=0)

    weights = np.asarray(sample_weight, dtype=np.float64)
    weights = weights / np.sum(weights)

    if vals.ndim == 1:
        sorter = np.argsort(vals)
        v_sorted = vals[sorter]
        w_sorted = weights[sorter]
        cdf = np.cumsum(w_sorted) - 0.5 * w_sorted
        cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
        return np.interp(q_arr, cdf, v_sorted)

    if vals.ndim == 2:
        _M, T = vals.shape
        out = np.zeros((len(q_arr), T), dtype=np.float64)
        for t in range(T):
            col = vals[:, t]
            sorter = np.argsort(col)
            v_sorted = col[sorter]
            w_sorted = weights[sorter]
            cdf = np.cumsum(w_sorted) - 0.5 * w_sorted
            cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
            out[:, t] = np.interp(q_arr, cdf, v_sorted)
        return out

    raise ValueError("`values` must be 1D or 2D.")


def compute_kde_1d(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    n_points: int = 250,
    padding: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a smooth 1D Kernel Density Estimate (supporting scenario probability weights).
    """
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    vals = vals[mask]
    if len(vals) < 2 or np.std(vals) < 1e-12:
        center = float(np.mean(vals)) if len(vals) > 0 else 0.0
        x_grid = np.linspace(center - 0.05, center + 0.05, n_points)
        y_grid = np.exp(-0.5 * ((x_grid - center) / 0.01) ** 2)
        return x_grid, y_grid

    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = max(vmax - vmin, 1e-6)
    x_grid = np.linspace(vmin - padding * span, vmax + padding * span, n_points)

    w_clean = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)[mask]
        if np.sum(w_arr) > 0:
            w_clean = w_arr / np.sum(w_arr)

    try:
        kde = gaussian_kde(vals, weights=w_clean)
        y_grid = kde(x_grid)
    except (ValueError, np.linalg.LinAlgError):
        hist, bin_edges = np.histogram(
            vals, bins=min(30, max(5, len(vals) // 5)), weights=w_clean, density=True
        )
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_grid = np.interp(x_grid, centers, hist, left=0.0, right=0.0)

    return x_grid, y_grid


def extract_equity_paths(
    source: MonteCarloResult | ScenarioMarketData | pd.DataFrame | np.ndarray,
    asset_weights: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, pd.Index]:
    """
    Extracts a 2D array of portfolio equity paths of shape (M, T_steps) and corresponding timestamps
    from a MonteCarloResult, ScenarioMarketData, DataFrame, or ndarray.
    """
    if isinstance(source, MonteCarloResult):
        # equity_curves in MonteCarloResult has shape (T, M) where columns are Path_0..Path_M-1
        df_eq = source.equity_curves
        return df_eq.to_numpy(dtype=np.float64).T, df_eq.index

    if isinstance(source, ScenarioMarketData):
        _M, _T, N = source.prices.shape
        if asset_weights is None:
            w = np.full(N, 1.0 / N, dtype=np.float64)
        elif isinstance(asset_weights, pd.Series):
            w = (
                asset_weights.reindex(source.asset_names)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            w = np.asarray(asset_weights, dtype=np.float64)

        if np.sum(np.abs(w)) > 0:
            w = w / np.sum(w)

        if source.initial_prices is not None:
            init_p = np.asarray(source.initial_prices, dtype=np.float64)[None, None, :]
            rel_prices = source.prices.astype(np.float64) / init_p
            step_wealth = np.asarray(
                np.tensordot(rel_prices, w, axes=([2], [0])), dtype=np.float64
            )  # (M, T)
        else:
            step_rets = np.tensordot(
                source.returns.astype(np.float64), w, axes=([2], [0])
            )
            step_wealth = np.asarray(
                np.cumprod(1.0 + step_rets, axis=1), dtype=np.float64
            )

        return step_wealth, source.timestamps

    if isinstance(source, pd.DataFrame):
        # Convention: if columns start with "Path" or index is DatetimeIndex, shape is (T, M)
        if isinstance(source.index, pd.DatetimeIndex) or any(
            str(c).startswith("Path") or str(c).startswith("CPCV")
            for c in source.columns
        ):
            return source.to_numpy(dtype=np.float64).T, source.index
        return source.to_numpy(dtype=np.float64), source.columns

    arr = np.asarray(source, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Equity paths array must be 2D of shape (M, T).")
    return arr, pd.RangeIndex(arr.shape[1])


def compute_drawdown_paths(equity_paths: np.ndarray) -> np.ndarray:
    """
    Computes underwater drawdown trajectories of shape (M, T) from equity paths of shape (M, T).
    Prepends 1.0 peak so initial drops below 1.0 are captured accurately.
    """
    eq = np.asarray(equity_paths, dtype=np.float64)
    running_peaks = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    drawdowns = (eq - running_peaks) / np.maximum(running_peaks, 1e-12)
    return drawdowns


def compute_risk_contributions(
    weights: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """
    Computes Capital Weight, Marginal Risk Contribution (MRC), Component Risk Contribution (CRC),
    and Percentage Component Risk Contribution (PCRC) for a portfolio.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        assets = list(cov_matrix.columns)
        Sigma = cov_matrix.to_numpy(dtype=np.float64)
    else:
        Sigma = np.asarray(cov_matrix, dtype=np.float64)
        assets = [f"Asset_{i}" for i in range(Sigma.shape[0])]

    if isinstance(weights, pd.Series):
        w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    port_var = float(w.T @ Sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 1e-16)))

    sigma_w = Sigma @ w
    mrc = sigma_w / port_vol
    crc = w * mrc
    pcrc = crc / port_vol if port_vol > 1e-12 else np.zeros_like(w)

    return pd.DataFrame(
        {
            "weight": w,
            "mrc": mrc,
            "crc": crc,
            "pcrc": pcrc,
        },
        index=assets,
    )


def reconstruct_drifted_weights(
    backtest_result: BacktestResult | pd.DataFrame,
    market_data: MarketData | pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstructs true intra-period drifted asset weights and rebalance turnover events.
    When `market_data` (or asset prices DataFrame) is provided, simulates natural market drift:
        w_{i, t}^- = w_{i, t-1}^+ * (1 + r_{i, t}) / sum_j w_{j, t-1}^+ * (1 + r_{j, t})
    and snaps back to target weights on dates where the target allocation changes.
    Returns:
        (drifted_weights_df, turnover_series)
    """
    if isinstance(backtest_result, BacktestResult):
        target_w = backtest_result.weights.dropna(how="all").fillna(0.0)
    else:
        target_w = backtest_result.dropna(how="all").fillna(0.0)

    if len(target_w) == 0:
        return target_w.copy(), pd.Series(dtype=float)

    prices_df: pd.DataFrame | None = None
    if isinstance(market_data, MarketData):
        prices_df = market_data.prices
    elif isinstance(market_data, pd.DataFrame):
        prices_df = market_data

    # Identify rebalance dates where target weights change (or first valid date)
    w_diff = target_w.diff().abs().sum(axis=1)
    rebalance_mask = (w_diff > 1e-7) | (np.arange(len(target_w)) == 0)

    if prices_df is None:
        turnover = w_diff.where(rebalance_mask, 0.0)
        return target_w.copy(), turnover

    common_dates = target_w.index.intersection(prices_df.index)
    if len(common_dates) < 2:
        turnover = w_diff.where(rebalance_mask, 0.0)
        return target_w.copy(), turnover

    target_aligned = target_w.loc[common_dates]
    asset_rets = (
        prices_df.reindex(columns=target_aligned.columns)
        .pct_change()
        .reindex(common_dates)
        .fillna(0.0)
    )
    reb_aligned = rebalance_mask.loc[common_dates]

    T, N = target_aligned.shape
    drifted_arr = np.zeros((T, N), dtype=np.float64)
    turnover_arr = np.zeros(T, dtype=np.float64)

    target_np = target_aligned.to_numpy(dtype=np.float64)
    rets_np = asset_rets.to_numpy(dtype=np.float64)
    reb_np = reb_aligned.to_numpy(dtype=bool)

    current_w = target_np[0].copy()
    drifted_arr[0] = current_w
    turnover_arr[0] = float(np.sum(np.abs(current_w)))

    for t in range(1, T):
        # Drift weights by day t's asset returns
        grown = current_w * (1.0 + rets_np[t])
        total_val = float(np.sum(grown))
        pre_rebalance_w = grown / total_val if total_val > 1e-12 else current_w

        if reb_np[t]:
            # Snap back to new target weights
            new_target = target_np[t]
            turnover_arr[t] = float(np.sum(np.abs(new_target - pre_rebalance_w)))
            current_w = new_target.copy()
        else:
            current_w = pre_rebalance_w

        drifted_arr[t] = current_w

    drifted_df = pd.DataFrame(
        drifted_arr, index=common_dates, columns=target_aligned.columns
    )
    turnover_series = pd.Series(turnover_arr, index=common_dates, name="Turnover")
    return drifted_df, turnover_series
