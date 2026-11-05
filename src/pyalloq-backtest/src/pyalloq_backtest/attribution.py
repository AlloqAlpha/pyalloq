from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from pyalloq_core.data import MarketData
from pyalloq_core.results import BacktestResult
from scipy.stats import t as student_t


@dataclass
class BrinsonAttributionResult:
    """
    Holds Classical Brinson-Fachler (1985) active return attribution across asset clusters/sectors.
    Decomposes active return (R_p - R_b) into:
      - Allocation Effect: (w_{p,j} - w_{b,j}) * (R_{b,j} - R_b)
      - Selection Effect:  w_{b,j} * (R_{p,j} - R_{b,j})
      - Interaction Effect: (w_{p,j} - w_{b,j}) * (R_{p,j} - R_{b,j})
    """

    sector_summary: pd.DataFrame
    timeseries: pd.DataFrame
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    total_active_return: float
    linking_method: str
    sector_map: dict[str, str]


@dataclass
class FactorAttributionResult:
    """
    Holds Factor-Implied Active Return Regression results (OLS with Newey-West HAC t-statistics).
    Decomposes active or excess return into systematic factor exposures (e.g., Market, Size,
    Value, Momentum, LowVolatility) and idiosyncratic Alpha.
    """

    factor_summary: pd.DataFrame
    r_squared: float
    adj_r_squared: float
    annualized_alpha: float
    rolling_exposures: pd.DataFrame
    cumulative_contributions: pd.DataFrame


@dataclass
class FrictionalDragResult:
    """
    Holds Frictional Drag Accounting (Ma & Smith, 2025; Almgren & Chriss, 2001) bridging
    Theoretical Paper Return -> Gross Executed Return -> Net Realized Return.
    """

    waterfall_summary: pd.Series
    daily_drag_df: pd.DataFrame
    cost_retention_ratio: float


@dataclass
class AttributionResult:
    """
    Unified container for Brinson-Fachler, Factor-Implied, and Frictional Drag attribution.
    """

    strategy_name: str
    brinson: BrinsonAttributionResult
    factor: FactorAttributionResult
    frictional: FrictionalDragResult

    def summary(self) -> pd.DataFrame:
        """Returns a consolidated high-level summary DataFrame of all three attribution pillars."""
        rows: dict[str, float] = {
            "brinson_allocation_effect": self.brinson.total_allocation_effect,
            "brinson_selection_effect": self.brinson.total_selection_effect,
            "brinson_interaction_effect": self.brinson.total_interaction_effect,
            "brinson_total_active_return": self.brinson.total_active_return,
            "factor_annualized_alpha": self.factor.annualized_alpha,
            "factor_r_squared": self.factor.r_squared,
            "factor_adj_r_squared": self.factor.adj_r_squared,
            "paper_gross_return": float(
                self.frictional.waterfall_summary["theoretical_paper_return"]
            ),
            "alpha_decay_and_drift_drag": float(
                self.frictional.waterfall_summary["alpha_decay_and_drift_drag"]
            ),
            "executed_gross_return": float(
                self.frictional.waterfall_summary["gross_executed_return"]
            ),
            "spread_and_commission_drag": float(
                self.frictional.waterfall_summary["spread_and_commission_drag"]
            ),
            "market_impact_drag": float(
                self.frictional.waterfall_summary["market_impact_drag"]
            ),
            "short_borrow_drag": float(
                self.frictional.waterfall_summary["short_borrow_drag"]
            ),
            "net_realized_return": float(
                self.frictional.waterfall_summary["net_realized_return"]
            ),
            "cost_retention_ratio": self.frictional.cost_retention_ratio,
        }
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=[self.strategy_name]
        )


class PerformanceAttributor:
    """
    Institutional Performance Attribution Engine for PyAlloq.

    Decomposes portfolio outperformance and execution mechanics into three pillars:
      1. Classical Brinson-Fachler (1985) Sector/Cluster Decomposition (with Carino 1999
         geometric or arithmetic multi-period linking and 3-tier cluster resolution).
      2. Factor-Implied Attribution (OLS regression with Newey-West HAC standard errors
         against explicit factor returns or auto-constructed cross-sectional risk premia).
      3. Frictional Drag Accounting (Ma & Smith, 2025) isolating alpha decay / intra-period
         weight drift from linear spread/commission and square-root market impact costs.
    """

    def __init__(
        self,
        linking_method: str = "carino",
        n_auto_clusters: int = 3,
        hac_lags: int = 5,
        rolling_factor_window: int = 60,
        ann_factor: float = 252.0,
    ) -> None:
        if linking_method not in ("carino", "arithmetic"):
            raise ValueError("`linking_method` must be 'carino' or 'arithmetic'.")
        self.linking_method = linking_method
        self.n_auto_clusters = max(2, int(n_auto_clusters))
        self.hac_lags = max(0, int(hac_lags))
        self.rolling_factor_window = max(10, int(rolling_factor_window))
        self.ann_factor = float(ann_factor)

    def resolve_sector_map(
        self,
        market_data: MarketData,
        assets: Sequence[str] | None = None,
        sector_map: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    ) -> dict[str, str]:
        """
        Resolves asset-to-sector/cluster mappings using a 3-tier hierarchy:
          1. Explicit `sector_map` argument (dict, Series, or DataFrame).
          2. `market_data.cross_sectional` ('sector', 'group', 'cluster', 'industry', 'class').
          3. Automatic hierarchical correlation clustering (Ward / correlation distance)
             mirroring HRP / HERC / NCO cluster formation.
        """
        asset_list = (
            [str(a) for a in assets]
            if assets is not None
            else [str(a) for a in market_data.prices.columns]
        )

        resolved: dict[str, str] = {}

        # Tier 1: Explicit sector_map
        if sector_map is not None:
            if isinstance(sector_map, pd.DataFrame):
                col = sector_map.columns[0]
                for a in asset_list:
                    if a in sector_map.index:
                        resolved[a] = str(sector_map.loc[a, col])
            elif isinstance(sector_map, pd.Series):
                for a in asset_list:
                    if a in sector_map.index:
                        resolved[a] = str(sector_map.loc[a])
            else:
                for a in asset_list:
                    if a in sector_map:
                        resolved[a] = str(sector_map[a])
            if resolved:
                for a in asset_list:
                    resolved.setdefault(a, "Other")
                return resolved

        # Tier 2: MarketData.cross_sectional metadata
        cs = market_data.cross_sectional
        if cs is not None and not cs.empty:
            candidate_cols = [
                c
                for c in cs.columns
                if any(
                    kw in str(c).lower()
                    for kw in ("sector", "group", "cluster", "industry", "class")
                )
            ]
            if candidate_cols:
                col = candidate_cols[0]
                for a in asset_list:
                    if a in cs.index:
                        resolved[a] = str(cs.loc[a, col])
                if resolved:
                    for a in asset_list:
                        resolved.setdefault(a, "Other")
                    return resolved

        # Tier 3: Automatic hierarchical correlation clustering
        n_assets = len(asset_list)
        if n_assets <= 2:
            return {a: f"Cluster_{idx + 1}" for idx, a in enumerate(asset_list)}

        rets = market_data.prices.reindex(columns=asset_list).pct_change().dropna()
        if len(rets) > 3:
            corr = rets.corr().fillna(0.0).to_numpy(dtype=np.float64, copy=True)
            np.fill_diagonal(corr, 1.0)
            dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0)).copy()
            np.fill_diagonal(dist, 0.0)
            try:
                condensed = ssd.squareform(dist, checks=False)
                link = sch.linkage(condensed, method="ward")
                k = min(self.n_auto_clusters, n_assets)
                labels = sch.fcluster(link, t=k, criterion="maxclust")
                return {
                    asset_list[i]: f"Cluster_{int(labels[i])}" for i in range(n_assets)
                }
            except (ValueError, np.linalg.LinAlgError):
                pass

        k = min(self.n_auto_clusters, n_assets)
        return {asset_list[i]: f"Cluster_{(i % k) + 1}" for i in range(n_assets)}

    def build_proxy_factors(
        self,
        market_data: MarketData,
        factor_returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Returns aligned daily factor return series for Factor-Implied Attribution.
        Uses explicit `factor_returns` or `market_data.features['factors']` when provided;
        otherwise constructs canonical cross-sectional / time-series risk premia proxies
        directly from `market_data` (`Market`, `Momentum`, `LowVolatility`, plus `Size`
        and `Value` if available in `market_data.cross_sectional`).
        """
        asset_rets = market_data.prices.pct_change().dropna()
        if factor_returns is not None and not factor_returns.empty:
            return factor_returns.reindex(asset_rets.index).fillna(0.0)

        if (
            "factors" in market_data.features
            and not market_data.features["factors"].empty
        ):
            return market_data.features["factors"].reindex(asset_rets.index).fillna(0.0)

        n_dates, n_assets = asset_rets.shape
        if n_dates == 0:
            return pd.DataFrame(
                columns=["Market", "Momentum", "LowVolatility"], index=asset_rets.index
            )

        rf_daily: pd.Series | float
        if isinstance(market_data.risk_free_rate, pd.Series):
            rf_daily = (
                market_data.risk_free_rate.reindex(asset_rets.index).ffill().fillna(0.0)
                / self.ann_factor
            )
        else:
            rf_daily = float(market_data.risk_free_rate) / self.ann_factor

        factors_dict: dict[str, pd.Series] = {}

        # 1. Market Excess Return (MKT)
        mkt_ret = asset_rets.mean(axis=1) - rf_daily
        factors_dict["Market"] = mkt_ret

        if n_assets >= 2:
            # 2. Cross-Sectional Momentum (Winners minus Losers using lagged trailing returns)
            mom_win = min(60, max(5, n_dates // 4))
            lagged_cum_ret = (
                (1.0 + asset_rets)
                .rolling(window=mom_win, min_periods=2)
                .apply(np.prod, raw=True)
                .shift(1)
                .fillna(1.0)
            )
            ranks_mom = lagged_cum_ret.rank(axis=1, pct=True)
            demeaned_mom = ranks_mom.sub(ranks_mom.mean(axis=1), axis=0)
            denom_mom = demeaned_mom.abs().sum(axis=1).replace(0.0, 1.0)
            w_mom = demeaned_mom.div(denom_mom, axis=0) * 2.0
            factors_dict["Momentum"] = (w_mom * asset_rets).sum(axis=1)

            # 3. Cross-Sectional Low Volatility (BAB / Low-Vol minus High-Vol)
            vol_win = min(20, max(5, n_dates // 5))
            lagged_vol = (
                asset_rets.rolling(window=vol_win, min_periods=2)
                .std()
                .shift(1)
                .bfill()
                .fillna(0.01)
            )
            ranks_low_vol = (-lagged_vol).rank(axis=1, pct=True)
            demeaned_lv = ranks_low_vol.sub(ranks_low_vol.mean(axis=1), axis=0)
            denom_lv = demeaned_lv.abs().sum(axis=1).replace(0.0, 1.0)
            w_lv = demeaned_lv.div(denom_lv, axis=0) * 2.0
            factors_dict["LowVolatility"] = (w_lv * asset_rets).sum(axis=1)

            # 4. Optional Size (SMB) and Value (HML) from cross_sectional metadata
            cs = market_data.cross_sectional
            if cs is not None and not cs.empty:
                for col in cs.columns:
                    col_lower = str(col).lower()
                    if any(k in col_lower for k in ("market_cap", "mcap", "size")):
                        vals = pd.to_numeric(
                            cs[col].reindex(asset_rets.columns), errors="coerce"
                        ).dropna()
                        if len(vals) >= 2:
                            # Small minus Big
                            r_smb = (-vals).rank(pct=True)
                            w_smb = r_smb - float(r_smb.mean())
                            norm_s = float(w_smb.abs().sum())
                            if norm_s > 1e-12:
                                w_smb = (w_smb / norm_s) * 2.0
                                factors_dict["Size"] = (
                                    asset_rets[vals.index]
                                    .mul(w_smb, axis=1)
                                    .sum(axis=1)
                                )
                    elif any(
                        k in col_lower for k in ("book_to_market", "btm", "value", "pe")
                    ):
                        vals = pd.to_numeric(
                            cs[col].reindex(asset_rets.columns), errors="coerce"
                        ).dropna()
                        if len(vals) >= 2:
                            sign = -1.0 if "pe" in col_lower else 1.0
                            r_val = (sign * vals).rank(pct=True)
                            w_val = r_val - float(r_val.mean())
                            norm_v = float(w_val.abs().sum())
                            if norm_v > 1e-12:
                                w_val = (w_val / norm_v) * 2.0
                                factors_dict["Value"] = (
                                    asset_rets[vals.index]
                                    .mul(w_val, axis=1)
                                    .sum(axis=1)
                                )
        else:
            factors_dict["Momentum"] = pd.Series(0.0, index=asset_rets.index)
            factors_dict["LowVolatility"] = pd.Series(0.0, index=asset_rets.index)

        return pd.DataFrame(factors_dict, index=asset_rets.index).fillna(0.0)

    def compute_brinson_fachler(
        self,
        backtest_result: BacktestResult,
        market_data: MarketData,
        benchmark_weights: pd.DataFrame | pd.Series | BacktestResult | None = None,
        sector_map: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
        linking_method: str | None = None,
    ) -> BrinsonAttributionResult:
        """
        Computes Classical Brinson-Fachler (1985) active return attribution across clusters/sectors.
        Supports both Carino (1999) geometric multi-period linking and arithmetic linking.
        """
        method = linking_method or self.linking_method
        if method not in ("carino", "arithmetic"):
            raise ValueError("`linking_method` must be 'carino' or 'arithmetic'.")

        port_w_df = backtest_result.weights.dropna(how="all").fillna(0.0)
        asset_rets_all = market_data.prices.pct_change().dropna()
        common_dates = port_w_df.index.intersection(asset_rets_all.index)
        assets = [str(c) for c in port_w_df.columns if c in asset_rets_all.columns]

        resolved_map = self.resolve_sector_map(
            market_data, assets=assets, sector_map=sector_map
        )
        sectors = sorted(set(resolved_map.values()))

        if len(common_dates) == 0 or len(assets) == 0:
            empty_summary = pd.DataFrame(
                0.0,
                index=sectors,
                columns=[
                    "portfolio_weight",
                    "benchmark_weight",
                    "active_weight",
                    "portfolio_return",
                    "benchmark_return",
                    "allocation_effect",
                    "selection_effect",
                    "interaction_effect",
                    "total_active_return",
                ],
            )
            empty_ts = pd.DataFrame(
                columns=[
                    "allocation_effect",
                    "selection_effect",
                    "interaction_effect",
                    "active_return",
                    "portfolio_return",
                    "benchmark_return",
                ]
            )
            return BrinsonAttributionResult(
                sector_summary=empty_summary,
                timeseries=empty_ts,
                total_allocation_effect=0.0,
                total_selection_effect=0.0,
                total_interaction_effect=0.0,
                total_active_return=0.0,
                linking_method=method,
                sector_map=resolved_map,
            )

        W_p = port_w_df.loc[common_dates, assets].to_numpy(dtype=np.float64)
        R_assets = asset_rets_all.loc[common_dates, assets].to_numpy(dtype=np.float64)
        T, N = W_p.shape

        # Normalize row sums of W_p if needed so Brinson identity holds cleanly
        row_sums_p = np.sum(W_p, axis=1, keepdims=True)
        W_p = np.where(np.abs(row_sums_p) > 1e-12, W_p / row_sums_p, 1.0 / N)

        # Resolve benchmark weights W_b of shape (T, N)
        if benchmark_weights is None:
            W_b = np.full((T, N), 1.0 / N, dtype=np.float64)
        elif isinstance(benchmark_weights, BacktestResult):
            bw_df = (
                benchmark_weights.weights.reindex(index=common_dates, columns=assets)
                .ffill()
                .bfill()
                .fillna(1.0 / N)
            )
            W_b = bw_df.to_numpy(dtype=np.float64)
        elif isinstance(benchmark_weights, pd.DataFrame):
            bw_df = (
                benchmark_weights.reindex(index=common_dates, columns=assets)
                .ffill()
                .bfill()
                .fillna(1.0 / N)
            )
            W_b = bw_df.to_numpy(dtype=np.float64)
        else:
            bw_ser = (
                pd.Series(benchmark_weights)
                .reindex(assets)
                .fillna(1.0 / N)
                .to_numpy(dtype=np.float64)
            )
            W_b = np.tile(bw_ser, (T, 1))

        row_sums_b = np.sum(W_b, axis=1, keepdims=True)
        W_b = np.where(np.abs(row_sums_b) > 1e-12, W_b / row_sums_b, 1.0 / N)

        n_sectors = len(sectors)
        # Binary membership matrix M of shape (N, n_sectors)
        M_sec = np.zeros((N, n_sectors), dtype=np.float64)
        for i, a in enumerate(assets):
            s_idx = sectors.index(resolved_map[a])
            M_sec[i, s_idx] = 1.0

        # Sector weights: (T, n_sectors)
        W_p_sec = W_p @ M_sec
        W_b_sec = W_b @ M_sec

        # Weighted asset returns within each sector: (T, n_sectors)
        num_p_sec = (W_p * R_assets) @ M_sec
        num_b_sec = (W_b * R_assets) @ M_sec

        # Sector return R_{p,j,t} and R_{b,j,t}
        # Note: When W_{p,j,t} == 0, set R_{p,j,t} = R_{b,j,t} so selection/interaction = 0
        R_b_sec = np.where(
            np.abs(W_b_sec) > 1e-12,
            num_b_sec / np.where(np.abs(W_b_sec) > 1e-12, W_b_sec, 1.0),
            0.0,
        )
        R_p_sec = np.where(
            np.abs(W_p_sec) > 1e-12,
            num_p_sec / np.where(np.abs(W_p_sec) > 1e-12, W_p_sec, 1.0),
            R_b_sec,
        )

        # Total portfolio and benchmark gross return per period t: (T,)
        R_p_t = np.sum(W_p_sec * R_p_sec, axis=1)
        R_b_t = np.sum(W_b_sec * R_b_sec, axis=1)

        # Classical Brinson-Fachler single-period effects: shape (T, n_sectors)
        A_t = (W_p_sec - W_b_sec) * (R_b_sec - R_b_t[:, None])
        S_t = W_b_sec * (R_p_sec - R_b_sec)
        I_t = (W_p_sec - W_b_sec) * (R_p_sec - R_b_sec)

        if method == "carino":
            # Carino (1999) logarithmic multi-period linking factors
            safe_rp = np.maximum(1.0 + R_p_t, 1e-8)
            safe_rb = np.maximum(1.0 + R_b_t, 1e-8)
            diff_t = R_p_t - R_b_t
            k_t = np.where(
                np.abs(diff_t) > 1e-10,
                (np.log(safe_rp) - np.log(safe_rb))
                / np.where(np.abs(diff_t) > 1e-10, diff_t, 1.0),
                1.0 / safe_rp,
            )

            cum_rp = float(np.prod(safe_rp) - 1.0)
            cum_rb = float(np.prod(safe_rb) - 1.0)
            total_active = cum_rp - cum_rb

            if abs(total_active) > 1e-10:
                K_total = (
                    math.log(max(1.0 + cum_rp, 1e-8))
                    - math.log(max(1.0 + cum_rb, 1e-8))
                ) / total_active
            else:
                K_total = 1.0 / max(1.0 + cum_rp, 1e-8)

            scale_t = k_t / max(K_total, 1e-12)
            # Enforce exact machine-precision summation if slight log-clipping occurred
            linked_sum = float(np.sum(scale_t * diff_t))
            if abs(linked_sum) > 1e-12 and abs(total_active) > 1e-12:
                scale_t = scale_t * (total_active / linked_sum)

            A_linked = A_t * scale_t[:, None]
            S_linked = S_t * scale_t[:, None]
            I_linked = I_t * scale_t[:, None]

            sec_p_ret = np.prod(1.0 + R_p_sec, axis=0) - 1.0
            sec_b_ret = np.prod(1.0 + R_b_sec, axis=0) - 1.0
        else:
            A_linked = A_t
            S_linked = S_t
            I_linked = I_t
            total_active = float(np.sum(R_p_t - R_b_t))
            sec_p_ret = np.sum(R_p_sec, axis=0)
            sec_b_ret = np.sum(R_b_sec, axis=0)

        sec_alloc = np.sum(A_linked, axis=0)
        sec_sel = np.sum(S_linked, axis=0)
        sec_inter = np.sum(I_linked, axis=0)
        sec_tot = sec_alloc + sec_sel + sec_inter

        avg_wp = np.mean(W_p_sec, axis=0)
        avg_wb = np.mean(W_b_sec, axis=0)

        sector_summary = pd.DataFrame(
            {
                "portfolio_weight": avg_wp,
                "benchmark_weight": avg_wb,
                "active_weight": avg_wp - avg_wb,
                "portfolio_return": sec_p_ret,
                "benchmark_return": sec_b_ret,
                "allocation_effect": sec_alloc,
                "selection_effect": sec_sel,
                "interaction_effect": sec_inter,
                "total_active_return": sec_tot,
            },
            index=sectors,
        )

        ts_alloc = np.sum(A_linked, axis=1)
        ts_sel = np.sum(S_linked, axis=1)
        ts_inter = np.sum(I_linked, axis=1)
        timeseries = pd.DataFrame(
            {
                "allocation_effect": ts_alloc,
                "selection_effect": ts_sel,
                "interaction_effect": ts_inter,
                "active_return": ts_alloc + ts_sel + ts_inter,
                "portfolio_return": R_p_t,
                "benchmark_return": R_b_t,
            },
            index=common_dates,
        )

        return BrinsonAttributionResult(
            sector_summary=sector_summary,
            timeseries=timeseries,
            total_allocation_effect=float(np.sum(sec_alloc)),
            total_selection_effect=float(np.sum(sec_sel)),
            total_interaction_effect=float(np.sum(sec_inter)),
            total_active_return=float(np.sum(sec_tot)),
            linking_method=method,
            sector_map=resolved_map,
        )

    @staticmethod
    def _ols_newey_west(
        y: np.ndarray, X: np.ndarray, max_lags: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Computes OLS coefficients with Newey-West (1987) HAC heteroskedasticity and
        autocorrelation-consistent standard errors, t-statistics, p-values, R^2, and Adj R^2.
        `X` should already include the leading constant intercept column of 1s.
        """
        T, K = X.shape
        xtx = X.T @ X
        try:
            xtx_inv = np.linalg.pinv(xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.eye(K) * 1e-6

        beta = xtx_inv @ (X.T @ y)
        resid = y - X @ beta

        tss = float(np.sum((y - np.mean(y)) ** 2))
        rss = float(np.sum(resid**2))
        r2 = float(np.clip(1.0 - rss / tss, 0.0, 1.0)) if tss > 1e-16 else 0.0
        dof = max(T - K, 1)
        adj_r2 = (
            float(1.0 - (1.0 - r2) * (T - 1) / dof) if T > K and tss > 1e-16 else r2
        )

        # Bartlett-kernel Newey-West sandwich covariance matrix
        S_hat = (X * (resid[:, None] ** 2)).T @ X
        L = min(max_lags, max(0, T // 4))
        for lag in range(1, L + 1):
            w_lag = 1.0 - lag / (L + 1.0)
            u_t = resid[lag:]
            u_lag = resid[:-lag]
            X_t = X[lag:]
            X_lag = X[:-lag]
            gamma_l = (X_t * (u_t * u_lag)[:, None]).T @ X_lag
            S_hat = S_hat + w_lag * (gamma_l + gamma_l.T)

        scale_dof = float(T) / float(dof)
        cov_beta = scale_dof * (xtx_inv @ S_hat @ xtx_inv)
        se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-16))
        t_stats = np.where(se > 1e-14, beta / se, 0.0)
        p_vals = 2.0 * (1.0 - student_t.cdf(np.abs(t_stats), df=dof))
        return beta, se, t_stats, p_vals, r2, adj_r2

    def compute_factor_attribution(
        self,
        backtest_result: BacktestResult | pd.Series,
        market_data: MarketData,
        benchmark_returns: pd.Series | BacktestResult | None = None,
        factor_returns: pd.DataFrame | None = None,
    ) -> FactorAttributionResult:
        """
        Regresses active return (R_p - R_b) or excess return against systematic risk premia
        (Market, Momentum, LowVolatility, Size, Value, or custom factors) using OLS + HAC.
        """
        port_ret = (
            backtest_result.returns.dropna()
            if isinstance(backtest_result, BacktestResult)
            else pd.Series(backtest_result).dropna()
        )
        factors_df = self.build_proxy_factors(
            market_data, factor_returns=factor_returns
        )

        common_dates = port_ret.index.intersection(factors_df.index)
        factor_cols = list(factors_df.columns)

        if len(common_dates) < max(5, len(factor_cols) + 2):
            idx_names = ["Alpha", *factor_cols]
            empty_df = pd.DataFrame(
                0.0,
                index=idx_names,
                columns=[
                    "exposure",
                    "std_error",
                    "t_stat",
                    "p_value",
                    "annualized_premia",
                    "annualized_contribution",
                    "pct_of_active_return",
                ],
            )
            return FactorAttributionResult(
                factor_summary=empty_df,
                r_squared=0.0,
                adj_r_squared=0.0,
                annualized_alpha=0.0,
                rolling_exposures=pd.DataFrame(columns=factor_cols, index=common_dates),
                cumulative_contributions=pd.DataFrame(
                    columns=idx_names, index=common_dates
                ),
            )

        y_port = port_ret.loc[common_dates].to_numpy(dtype=np.float64)
        if benchmark_returns is not None:
            b_ser = (
                benchmark_returns.returns
                if isinstance(benchmark_returns, BacktestResult)
                else pd.Series(benchmark_returns)
            )
            b_aligned = (
                b_ser.reindex(common_dates).fillna(0.0).to_numpy(dtype=np.float64)
            )
            y_target = y_port - b_aligned
        else:
            if isinstance(market_data.risk_free_rate, pd.Series):
                rf_arr = (
                    market_data.risk_free_rate.reindex(common_dates)
                    .ffill()
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                    / self.ann_factor
                )
            else:
                rf_arr = np.full(
                    len(common_dates),
                    float(market_data.risk_free_rate) / self.ann_factor,
                )
            y_target = y_port - rf_arr

        F_mat = factors_df.loc[common_dates, factor_cols].to_numpy(dtype=np.float64)
        T = len(common_dates)
        X_design = np.column_stack([np.ones(T, dtype=np.float64), F_mat])

        beta, se, t_stats, p_vals, r2, adj_r2 = self._ols_newey_west(
            y_target, X_design, max_lags=self.hac_lags
        )

        alpha_daily = float(beta[0])
        ann_alpha = alpha_daily * self.ann_factor
        factor_betas = beta[1:]

        ann_factor_premia = np.mean(F_mat, axis=0) * self.ann_factor
        ann_factor_contrib = factor_betas * ann_factor_premia

        all_exposures = np.concatenate([[alpha_daily], factor_betas])
        all_premia = np.concatenate([[self.ann_factor], ann_factor_premia])
        all_contrib = np.concatenate([[ann_alpha], ann_factor_contrib])

        total_ann_active = float(np.mean(y_target) * self.ann_factor)
        denom_active = total_ann_active if abs(total_ann_active) > 1e-10 else 1.0
        pct_active = all_contrib / denom_active

        idx_names = ["Alpha", *factor_cols]
        factor_summary = pd.DataFrame(
            {
                "exposure": all_exposures,
                "std_error": se,
                "t_stat": t_stats,
                "p_value": p_vals,
                "annualized_premia": all_premia,
                "annualized_contribution": all_contrib,
                "pct_of_active_return": pct_active,
            },
            index=idx_names,
        )

        # Daily cumulative factor return contributions
        daily_contrib_mat = np.column_stack(
            [
                np.full(T, alpha_daily),
                F_mat * factor_betas[None, :],
            ]
        )
        cum_contrib_df = pd.DataFrame(
            np.cumsum(daily_contrib_mat, axis=0),
            index=common_dates,
            columns=idx_names,
        )

        # Rolling factor exposures
        win = min(self.rolling_factor_window, max(10, T // 2))
        rolling_rows: list[np.ndarray] = []
        for t_idx in range(T):
            start_i = max(0, t_idx - win + 1)
            if t_idx - start_i + 1 < len(factor_cols) + 2:
                rolling_rows.append(factor_betas.copy())
            else:
                X_sub = X_design[start_i : t_idx + 1]
                y_sub = y_target[start_i : t_idx + 1]
                b_sub = np.linalg.pinv(X_sub) @ y_sub
                rolling_rows.append(b_sub[1:])

        rolling_exposures = pd.DataFrame(
            rolling_rows, index=common_dates, columns=factor_cols
        )

        return FactorAttributionResult(
            factor_summary=factor_summary,
            r_squared=r2,
            adj_r_squared=adj_r2,
            annualized_alpha=ann_alpha,
            rolling_exposures=rolling_exposures,
            cumulative_contributions=cum_contrib_df,
        )

    def compute_frictional_drag(
        self,
        backtest_result: BacktestResult,
        market_data: MarketData | None = None,
    ) -> FrictionalDragResult:
        """
        Decomposes the Gross-to-Net execution bridge (Ma & Smith, 2025):
          1. Theoretical Paper Return (un-drifted target weights)
          2. Alpha Decay & Drift / Threshold Delay Drag (R_executed_gross - R_paper)
          3. Gross Executed Return (R_executed_gross)
          4. Spread & Commission Drag (Linear L1 cost)
          5. Square-Root Market Impact Drag (Almgren-Chriss impact cost)
          6. Short Borrow Drag (Financing cost)
          7. Net Realized Return (R_net)
        """
        net_ret = backtest_result.returns.dropna()
        idx = net_ret.index

        if len(idx) == 0:
            empty_ser = pd.Series(
                0.0,
                index=[
                    "theoretical_paper_return",
                    "alpha_decay_and_drift_drag",
                    "gross_executed_return",
                    "spread_and_commission_drag",
                    "market_impact_drag",
                    "short_borrow_drag",
                    "total_explicit_cost_drag",
                    "net_realized_return",
                    "annualized_paper_return",
                    "annualized_gross_return",
                    "annualized_net_return",
                    "cost_retention_ratio",
                ],
            )
            return FrictionalDragResult(
                waterfall_summary=empty_ser,
                daily_drag_df=pd.DataFrame(),
                cost_retention_ratio=1.0,
            )

        costs_aligned = (
            backtest_result.costs.reindex(idx).fillna(0.0)
            if backtest_result.costs is not None
            else pd.Series(0.0, index=idx)
        )

        if backtest_result.gross_returns is not None:
            gross_ret = backtest_result.gross_returns.reindex(idx).fillna(
                net_ret + costs_aligned
            )
        else:
            gross_ret = net_ret + costs_aligned

        if backtest_result.paper_returns is not None:
            paper_ret = backtest_result.paper_returns.reindex(idx).fillna(gross_ret)
        elif (
            market_data is not None
            and backtest_result.target_weights is not None
            and not backtest_result.target_weights.empty
        ):
            asset_rets = market_data.prices.pct_change().reindex(idx).fillna(0.0)
            tw = (
                backtest_result.target_weights.reindex(
                    index=idx, columns=asset_rets.columns
                )
                .ffill()
                .fillna(0.0)
            )
            paper_ret = (tw * asset_rets).sum(axis=1)
        else:
            paper_ret = gross_ret.copy()

        if (
            backtest_result.cost_breakdown is not None
            and not backtest_result.cost_breakdown.empty
        ):
            cb = backtest_result.cost_breakdown.reindex(idx).fillna(0.0)
            spread_s = cb.get("spread_cost", costs_aligned)
            impact_s = cb.get("impact_cost", pd.Series(0.0, index=idx))
            borrow_s = cb.get("borrow_cost", pd.Series(0.0, index=idx))
        else:
            spread_s = costs_aligned.copy()
            impact_s = pd.Series(0.0, index=idx)
            borrow_s = pd.Series(0.0, index=idx)

        drift_drag_daily = gross_ret - paper_ret
        daily_df = pd.DataFrame(
            {
                "paper_return": paper_ret,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "alpha_decay_drift_drag": drift_drag_daily,
                "spread_cost": spread_s,
                "impact_cost": impact_s,
                "borrow_cost": borrow_s,
                "total_cost": costs_aligned,
            },
            index=idx,
        )

        # Arithmetic cumulative totals so the waterfall sums identically:
        # paper + drift_drag - spread - impact - borrow == net
        tot_paper = float(paper_ret.sum())
        tot_drift = float(drift_drag_daily.sum())
        tot_gross = float(gross_ret.sum())
        tot_spread = float(spread_s.sum())
        tot_impact = float(impact_s.sum())
        tot_borrow = float(borrow_s.sum())
        tot_cost = tot_spread + tot_impact + tot_borrow
        tot_net = float(net_ret.sum())

        n_years = max(len(idx) / self.ann_factor, 1e-4)
        cum_paper = float(np.prod(1.0 + paper_ret.to_numpy(dtype=np.float64)) - 1.0)
        cum_gross = float(np.prod(1.0 + gross_ret.to_numpy(dtype=np.float64)) - 1.0)
        cum_net = float(np.prod(1.0 + net_ret.to_numpy(dtype=np.float64)) - 1.0)

        ann_paper = float((1.0 + max(cum_paper, -0.9999)) ** (1.0 / n_years) - 1.0)
        ann_gross = float((1.0 + max(cum_gross, -0.9999)) ** (1.0 / n_years) - 1.0)
        ann_net = float((1.0 + max(cum_net, -0.9999)) ** (1.0 / n_years) - 1.0)

        retention = (
            float(tot_net / tot_gross)
            if abs(tot_gross) > 1e-8 and tot_gross > 0
            else (1.0 if tot_cost <= 1e-12 else 0.0)
        )

        summary_ser = pd.Series(
            {
                "theoretical_paper_return": tot_paper,
                "alpha_decay_and_drift_drag": tot_drift,
                "gross_executed_return": tot_gross,
                "spread_and_commission_drag": -tot_spread,
                "market_impact_drag": -tot_impact,
                "short_borrow_drag": -tot_borrow,
                "total_explicit_cost_drag": -tot_cost,
                "net_realized_return": tot_net,
                "annualized_paper_return": ann_paper,
                "annualized_gross_return": ann_gross,
                "annualized_net_return": ann_net,
                "cost_retention_ratio": retention,
            },
            name=backtest_result.name,
        )

        return FrictionalDragResult(
            waterfall_summary=summary_ser,
            daily_drag_df=daily_df,
            cost_retention_ratio=retention,
        )

    def attribute(
        self,
        backtest_result: BacktestResult,
        market_data: MarketData,
        benchmark_weights: pd.DataFrame | pd.Series | BacktestResult | None = None,
        benchmark_returns: pd.Series | BacktestResult | None = None,
        sector_map: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
        factor_returns: pd.DataFrame | None = None,
        linking_method: str | None = None,
    ) -> AttributionResult:
        """
        Runs the complete 3-pillar Performance Attribution suite (Brinson-Fachler,
        Factor-Implied Regression, and Frictional Drag Accounting) for a `BacktestResult`.
        """
        brinson_res = self.compute_brinson_fachler(
            backtest_result=backtest_result,
            market_data=market_data,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            linking_method=linking_method,
        )

        # If benchmark_returns is not explicitly provided, use the Brinson benchmark return series
        resolved_bench_rets: pd.Series | BacktestResult | None = benchmark_returns
        if resolved_bench_rets is None and not brinson_res.timeseries.empty:
            resolved_bench_rets = brinson_res.timeseries["benchmark_return"]

        factor_res = self.compute_factor_attribution(
            backtest_result=backtest_result,
            market_data=market_data,
            benchmark_returns=resolved_bench_rets,
            factor_returns=factor_returns,
        )

        frictional_res = self.compute_frictional_drag(
            backtest_result=backtest_result,
            market_data=market_data,
        )

        return AttributionResult(
            strategy_name=backtest_result.name,
            brinson=brinson_res,
            factor=factor_res,
            frictional=frictional_res,
        )
