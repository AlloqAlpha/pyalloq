import pandas as pd
from typing import Any, cast
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_backtest.splitters import BaseWindowSplitter, RollingWindowSplitter
from pyalloq_backtest.costs import BaseCostModel, FlatBpsCostModel
from pyalloq_backtest.metrics import MetricsTearSheet
from pyalloq_core.data import MarketData
from pyalloq_backtest.costs import CostData


class WalkForwardEngine:
    def __init__(
        self,
        pipeline: StrategyPipeline,
        splitter: BaseWindowSplitter | None = None,
        cost_model: BaseCostModel | None = None,
        rebalance_freq: str = "ME",
    ) -> None:
        self.pipeline = pipeline
        self.rebalance_freq = rebalance_freq

        self.splitter = splitter or RollingWindowSplitter(lookback_window=252)
        self.cost_model = cost_model or FlatBpsCostModel(bps=10.0)

    def run(
        self,
        data: MarketData,
    ) -> dict[str, Any]:
        asset_returns = data.prices.pct_change().dropna()

        raw_index = data.prices.resample(self.rebalance_freq).last().index
        rebalance_dates = pd.DatetimeIndex(raw_index)

        weights_history = []
        for current_date, data_window in self.splitter.split(data, rebalance_dates):
            weights = self.pipeline.generate_weights(data_window)
            weights.name = current_date
            weights_history.append(weights)

        df_weights = pd.DataFrame(weights_history)
        df_weights_daily = df_weights.reindex(data.prices.index).ffill().shift(1)

        weight_changes = df_weights.diff().fillna(df_weights)
        turnover_costs_daily = pd.Series(0.0, index=data.prices.index)

        for raw_date, row in weight_changes.iterrows():
            date = pd.Timestamp(cast(Any, raw_date))

            if date in data.prices.index:
                daily_feats: dict[str, CostData] | None = None

                if data.features:
                    daily_feats = {}
                    for k, v in data.features.items():
                        daily_feats[k] = v.loc[date]

                costs = self.cost_model.calculate_costs(row, features_slice=daily_feats)
                turnover_costs_daily.loc[date] = costs.sum()  # type: ignore[call-overload]

        portfolio_returns = (df_weights_daily * asset_returns).sum(
            axis=1
        ) - turnover_costs_daily
        portfolio_returns = portfolio_returns.dropna()

        tear_sheet = MetricsTearSheet.generate(portfolio_returns)

        return {
            "returns": portfolio_returns,
            "weights": df_weights_daily,
            "tear_sheet": tear_sheet,
        }
