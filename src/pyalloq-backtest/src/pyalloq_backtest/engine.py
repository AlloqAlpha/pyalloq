from typing import Any, cast

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult

from pyalloq_backtest.costs import BaseCostModel, CostData, FlatBpsCostModel
from pyalloq_backtest.metrics import MetricsTearSheet
from pyalloq_backtest.splitters import BaseWindowSplitter, RollingWindowSplitter


class WalkForwardEngine:
    def __init__(
        self,
        pipeline: StrategyPipeline,
        splitter: BaseWindowSplitter | None = None,
        cost_model: BaseCostModel | None = None,
        rebalance_freq: str = "ME",
        drift_weights: bool = True,
        rebalance_threshold: float | None = None,
        short_borrow_bps: float = 0.0,
        max_gross_leverage: float | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.rebalance_freq = rebalance_freq
        self.drift_weights = drift_weights
        self.rebalance_threshold = rebalance_threshold
        self.short_borrow_bps = short_borrow_bps
        self.max_gross_leverage = max_gross_leverage
        self.benchmark_returns = benchmark_returns

        self.splitter = splitter or RollingWindowSplitter(lookback_window=252)
        self.cost_model = cost_model or FlatBpsCostModel(bps=10.0)

    def _resolve_rebalance_dates(self, prices: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Maps calendar resample periods to the actual last trading timestamp in `prices.index`
        within each bucket so weekend/holiday period-ends are never silently dropped.
        """
        idx_series = pd.Series(prices.index, index=prices.index)
        resampled = idx_series.resample(self.rebalance_freq).last().dropna()
        return pd.DatetimeIndex(resampled.values).unique()

    def _extract_features_slice(
        self, data: MarketData, date: pd.Timestamp
    ) -> dict[str, CostData] | None:
        if not data.features:
            return None
        daily_feats: dict[str, CostData] = {}
        for k, v in data.features.items():
            if date in v.index:
                daily_feats[k] = v.loc[date]
        return daily_feats

    def run(
        self,
        data: MarketData,
    ) -> BacktestResult:
        asset_returns = data.prices.pct_change().dropna()
        rebalance_dates = self._resolve_rebalance_dates(data.prices)

        split_windows: dict[pd.Timestamp, MarketData] = {
            pd.Timestamp(cast(Any, current_date)): data_window
            for current_date, data_window in self.splitter.split(data, rebalance_dates)
        }

        if not split_windows:
            empty_ret = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            empty_w = pd.DataFrame(columns=data.prices.columns, index=data.prices.index)
            tear_sheet = MetricsTearSheet.generate(
                empty_ret, risk_free_rate=data.risk_free_rate
            )
            return BacktestResult(
                name=self.pipeline.allocator.__class__.__name__,
                returns=empty_ret,
                weights=empty_w,
                tear_sheet=tear_sheet,
                target_weights=empty_w,
                turnover=pd.Series(0.0, index=data.prices.index),
                costs=pd.Series(0.0, index=data.prices.index),
            )

        all_dates = data.prices.index
        assets = list(data.prices.columns)
        n_dates = len(all_dates)
        n_assets = len(assets)

        # Daily risk-free rate for uninvested cash balance
        if isinstance(data.risk_free_rate, pd.Series):
            rf_daily = (
                data.risk_free_rate.reindex(all_dates).ffill().fillna(0.0).to_numpy()
                / 252.0
            )
        else:
            rf_daily = np.full(n_dates, float(data.risk_free_rate) / 252.0)

        rets_aligned = data.prices.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        weights_history: list[pd.Series] = []

        held_weights_arr = np.full((n_dates, n_assets), np.nan, dtype=np.float64)
        turnover_arr = np.zeros(n_dates, dtype=np.float64)
        costs_arr = np.zeros(n_dates, dtype=np.float64)
        gross_port_ret_arr = np.full(n_dates, np.nan, dtype=np.float64)

        active = False
        current_post_rebalance_w = np.zeros(n_assets, dtype=np.float64)
        daily_borrow_rate = (self.short_borrow_bps / 10000.0) / 252.0

        for t_idx, raw_dt in enumerate(all_dates):
            dt = pd.Timestamp(raw_dt)
            if active:
                # 1. Hold `current_post_rebalance_w` entering day t_idx
                held_w = current_post_rebalance_w.copy()
                held_weights_arr[t_idx] = held_w

                asset_r = rets_aligned[t_idx]
                cash_w = max(0.0, 1.0 - float(np.sum(held_w)))
                short_exposure = float(np.sum(np.maximum(0.0, -held_w)))
                borrow_cost = short_exposure * daily_borrow_rate

                gross_r = float(np.dot(held_w, asset_r)) + cash_w * float(
                    rf_daily[t_idx]
                )
                gross_port_ret_arr[t_idx] = gross_r
                costs_arr[t_idx] += borrow_cost

                # 2. Drift weights to end of day t_idx (pre-rebalance w_t^-)
                if self.drift_weights:
                    denom = (
                        1.0
                        + float(np.dot(held_w, asset_r))
                        + cash_w * float(rf_daily[t_idx])
                    )
                    if abs(denom) > 1e-12:
                        pre_reb_w = (held_w * (1.0 + asset_r)) / denom
                    else:
                        pre_reb_w = held_w.copy()
                else:
                    pre_reb_w = held_w.copy()
            else:
                pre_reb_w = np.zeros(n_assets, dtype=np.float64)

            # 3. Check if dt is a scheduled rebalance date
            if dt in split_windows:
                data_window = split_windows[dt]
                current_w_series = (
                    pd.Series(pre_reb_w, index=assets) if active else None
                )
                weights = self.pipeline.generate_weights(
                    data_window, current_weights=current_w_series
                ).copy()
                weights = weights.reindex(assets, fill_value=0.0)
                if self.max_gross_leverage is not None:
                    gross_lev = float(weights.abs().sum())
                    if gross_lev > self.max_gross_leverage and gross_lev > 1e-12:
                        weights = weights * (self.max_gross_leverage / gross_lev)
                weights.name = dt
                weights_history.append(weights)

                target_w = weights.to_numpy(dtype=np.float64)
                delta_w = target_w - pre_reb_w
                l1_drift = float(np.sum(np.abs(delta_w)))

                should_rebalance = (
                    (not active)
                    or (self.rebalance_threshold is None)
                    or (l1_drift >= self.rebalance_threshold)
                )

                if should_rebalance:
                    delta_series = pd.Series(delta_w, index=assets)
                    feats_slice = self._extract_features_slice(data, dt)
                    step_costs = self.cost_model.calculate_costs(
                        delta_series, features_slice=feats_slice
                    )
                    cost_val = float(pd.Series(step_costs).sum())
                    turnover_arr[t_idx] = l1_drift
                    costs_arr[t_idx] += cost_val
                    current_post_rebalance_w = target_w.copy()
                    active = True
                else:
                    current_post_rebalance_w = pre_reb_w.copy()
            elif active:
                current_post_rebalance_w = pre_reb_w.copy()

        df_rebalance_targets = pd.DataFrame(weights_history).reindex(
            columns=data.prices.columns, fill_value=0.0
        )
        df_target_daily = (
            df_rebalance_targets.reindex(data.prices.index).ffill().shift(1)
        )

        df_weights_daily = pd.DataFrame(
            held_weights_arr, index=all_dates, columns=assets
        )
        turnover_series = pd.Series(turnover_arr, index=all_dates, name="Turnover")
        costs_series = pd.Series(costs_arr, index=all_dates, name="Costs")

        net_returns_arr = gross_port_ret_arr - costs_arr
        portfolio_returns = pd.Series(net_returns_arr, index=all_dates).reindex(
            asset_returns.index
        )
        portfolio_returns = portfolio_returns.dropna()

        tear_sheet = MetricsTearSheet.generate(
            portfolio_returns,
            risk_free_rate=data.risk_free_rate,
            weights=df_weights_daily.reindex(portfolio_returns.index),
            turnover=turnover_series.reindex(portfolio_returns.index),
            benchmark_returns=self.benchmark_returns,
        )

        return BacktestResult(
            name=self.pipeline.allocator.__class__.__name__,
            returns=portfolio_returns,
            weights=df_weights_daily,
            tear_sheet=tear_sheet,
            target_weights=df_target_daily,
            turnover=turnover_series,
            costs=costs_series,
        )
