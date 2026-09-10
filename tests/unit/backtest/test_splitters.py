import pandas as pd
from pyalloq_backtest.splitters import ExpandingWindowSplitter, RollingWindowSplitter
from pyalloq_core.data import MarketData


class TestSplitters:
    def test_rolling_window_splitter_no_lookahead(
        self, synthetic_market_data: MarketData
    ) -> None:
        rebalance_dates = pd.DatetimeIndex(
            synthetic_market_data.prices.resample("ME").last().index
        )
        lookback = 60
        splitter = RollingWindowSplitter(lookback_window=lookback)

        splits = list(splitter.split(synthetic_market_data, rebalance_dates))
        assert len(splits) > 0

        for current_date, window in splits:
            # 1. Exact lookback length
            assert len(window.prices) == lookback
            # 2. Strict Point-In-Time property: window prices index ends at current_date or latest available before current_date
            assert window.prices.index[-1] <= current_date
            # 3. No lookahead: all dates in window must be <= current_date
            assert (window.prices.index <= current_date).all()

    def test_expanding_window_splitter(self, synthetic_market_data: MarketData) -> None:
        rebalance_dates = pd.DatetimeIndex(
            synthetic_market_data.prices.resample("ME").last().index
        )
        min_periods = 60
        splitter = ExpandingWindowSplitter(min_periods=min_periods)

        splits = list(splitter.split(synthetic_market_data, rebalance_dates))
        assert len(splits) > 0

        prev_len = 0
        for current_date, window in splits:
            assert len(window.prices) >= min_periods
            assert len(window.prices) >= prev_len  # Expanding
            prev_len = len(window.prices)
            assert (window.prices.index <= current_date).all()

    def test_purged_kfold_splitter(self, synthetic_market_data: MarketData) -> None:
        from pyalloq_backtest.splitters import PurgedKFoldSplitter

        splitter = PurgedKFoldSplitter(n_splits=3, embargo_pct=0.02)
        splits = list(splitter.split(synthetic_market_data))

        assert len(splits) == 3
        for train_data, test_data in splits:
            assert isinstance(train_data, MarketData)
            assert isinstance(test_data, MarketData)
            assert len(train_data.prices) > 0
            assert len(test_data.prices) > 0
            # Test that train and test indices have no overlap (purging)
            train_idx = set(train_data.prices.index)
            test_idx = set(test_data.prices.index)
            assert len(train_idx.intersection(test_idx)) == 0
