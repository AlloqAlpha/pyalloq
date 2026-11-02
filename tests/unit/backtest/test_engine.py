import numpy as np
import pandas as pd
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult, OptimizationResult

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.naive import EqualWeightAllocator


class PartialCashOrShortAllocator(BaseAllocator):
    """Helper allocator for testing cash accrual, short borrow, and leverage caps."""

    def __init__(self, tickers: list[str], target_weights: dict[str, float]) -> None:
        super().__init__(tickers)
        self.target_weights = target_weights

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs,
    ) -> OptimizationResult:
        w = pd.Series(self.target_weights).reindex(data.prices.columns, fill_value=0.0)
        return OptimizationResult(weights=w, status="OPTIMAL", name="CustomAlloc")


class TestBacktestEngine:
    def test_walk_forward_engine_execution(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        pipeline = StrategyPipeline(
            allocator=EqualWeightAllocator(tickers=sample_assets),
            returns_estimator=EWMAReturnEstimator(span=20),
            cov_estimator=EmpiricalCovariance(),
        )

        splitter = RollingWindowSplitter(lookback_window=60)
        cost_model = FlatBpsCostModel(bps=5.0)

        engine = WalkForwardEngine(
            pipeline=pipeline,
            splitter=splitter,
            cost_model=cost_model,
            rebalance_freq="ME",
        )

        results = engine.run(synthetic_market_data)

        # 1. Output structure
        assert isinstance(results, BacktestResult)
        assert "tear_sheet" in results
        assert "weights" in results
        assert "returns" in results
        assert results.target_weights is not None
        assert results.turnover is not None
        assert results.costs is not None

        # 2. Tear sheet metrics exist
        tear_sheet = results["tear_sheet"]
        assert results.tear_sheet is tear_sheet
        assert isinstance(tear_sheet, pd.DataFrame)
        assert "annualized_return" in tear_sheet.index
        assert "maximum_drawdown" in tear_sheet.index
        assert "sharpe_ratio" in tear_sheet.index
        assert "annualized_turnover" in tear_sheet.index
        assert "effective_number_of_bets" in tear_sheet.index

        # 3. Weights history matches assets and sums to ~1.0 on all active days
        weights_df = results["weights"].dropna()
        assert list(weights_df.columns) == sample_assets
        for _, row in weights_df.iterrows():
            assert np.isclose(row.sum(), 1.0, atol=1e-5)

        # 4. Portfolio returns length matches rebalanced test period
        port_returns = results["returns"]
        assert len(port_returns) > 0
        assert not port_returns.isna().any()

    def test_intra_period_weight_drift_and_weekend_month_ends(self) -> None:
        """
        Regression test:
        1. Calendar month-ends falling on weekends (e.g. 2023-04-30 Sunday) must map to the
           actual last business day (2023-04-28 Friday) and NEVER be dropped.
        2. Between rebalances, an Equal-Weight portfolio's weights must drift with diverging
           asset returns and incur non-zero rebalance turnover at subsequent month-ends.
        """
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        # Asset A trends up +1%/day, Asset B trends down -0.5%/day
        n = len(dates)
        prices = pd.DataFrame(
            {
                "A": 100.0 * (1.01 ** np.arange(n)),
                "B": 100.0 * (0.995 ** np.arange(n)),
            },
            index=dates,
        )
        md = MarketData(prices=prices)
        pipe = StrategyPipeline(allocator=EqualWeightAllocator(tickers=["A", "B"]))

        engine_drifted = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=20),
            cost_model=FlatBpsCostModel(bps=10.0),
            rebalance_freq="ME",
            drift_weights=True,
        )
        res_drifted = engine_drifted.run(md)

        # April 2023 ended on Sunday Apr 30; last business day was Friday 2023-04-28
        apr_last_bday = pd.Timestamp("2023-04-28")
        assert apr_last_bday in prices.index
        assert res_drifted.turnover is not None
        assert res_drifted.turnover.loc[apr_last_bday] > 0.0

        # Between rebalances, Asset A's weight must drift above 0.50
        w_valid = res_drifted.weights.dropna()
        assert (w_valid["A"] > 0.505).any()
        # The day after April rebalance (2023-05-01), weights must snap back to [0.5, 0.5]
        may_first_bday = pd.Timestamp("2023-05-01")
        np.testing.assert_allclose(
            w_valid.loc[may_first_bday].values, [0.5, 0.5], atol=1e-6
        )

    def test_rebalance_threshold_skips_small_trades(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        pipe = StrategyPipeline(allocator=EqualWeightAllocator(tickers=sample_assets))
        # High threshold (50% L1 drift) skips intermediate monthly rebalances
        engine_high_thresh = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=FlatBpsCostModel(bps=10.0),
            rebalance_freq="ME",
            rebalance_threshold=0.50,
        )
        engine_no_thresh = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=60),
            cost_model=FlatBpsCostModel(bps=10.0),
            rebalance_freq="ME",
            rebalance_threshold=None,
        )

        res_high = engine_high_thresh.run(synthetic_market_data)
        res_none = engine_no_thresh.run(synthetic_market_data)

        assert res_high.turnover is not None and res_none.turnover is not None
        assert res_high.turnover.sum() < res_none.turnover.sum()

    def test_cash_return_short_borrow_and_leverage_cap(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=60)
        prices = pd.DataFrame(
            {"A": np.full(60, 100.0), "B": np.full(60, 100.0)}, index=dates
        )
        # 5.04% risk free rate, flat asset prices
        md = MarketData(prices=prices, risk_free_rate=0.0504)

        # 1. 50% in A, 0% in B -> 50% uninvested cash earning risk_free_rate
        cash_alloc = PartialCashOrShortAllocator(["A", "B"], {"A": 0.5, "B": 0.0})
        engine_cash = WalkForwardEngine(
            pipeline=StrategyPipeline(allocator=cash_alloc),
            splitter=RollingWindowSplitter(lookback_window=10),
            cost_model=FlatBpsCostModel(bps=0.0),
            rebalance_freq="ME",
        )
        res_cash = engine_cash.run(md)
        # Daily return on day 1 is 0.5 * (0.0504 / 252) = 0.0001, and ~0.0001 across the month as cash compounds
        assert np.isclose(res_cash.returns.iloc[0], 0.0001, rtol=1e-6)
        np.testing.assert_allclose(res_cash.returns.values, 0.0001, atol=1e-6)

        # 2. Long-short with max_gross_leverage cap and short_borrow_bps
        ls_alloc = PartialCashOrShortAllocator(["A", "B"], {"A": 1.5, "B": -0.5})
        engine_ls = WalkForwardEngine(
            pipeline=StrategyPipeline(allocator=ls_alloc),
            splitter=RollingWindowSplitter(lookback_window=10),
            cost_model=FlatBpsCostModel(bps=0.0),
            rebalance_freq="ME",
            max_gross_leverage=1.0,  # Scales [1.5, -0.5] (gross 2.0) down by 0.5 -> [0.75, -0.25]
            short_borrow_bps=252.0,  # 1 bp per day on short leg (0.25 * 0.0001 = 0.000025/day)
        )
        res_ls = engine_ls.run(md)
        w_ls = res_ls.weights.dropna()
        np.testing.assert_allclose(w_ls.iloc[0].values, [0.75, -0.25], rtol=1e-5)
        assert res_ls.costs is not None
        assert (res_ls.costs.reindex(res_ls.returns.index) > 0.0).all()
