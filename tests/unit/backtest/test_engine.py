import numpy as np
import pandas as pd
from pyalloq_backtest.costs import FlatBpsCostModel
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.pipeline import StrategyPipeline

from pyalloq.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.optimizers.classical.naive import EqualWeightAllocator


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
        assert "tear_sheet" in results
        assert "weights" in results
        assert "returns" in results

        # 2. Tear sheet metrics exist
        tear_sheet = results["tear_sheet"]
        assert isinstance(tear_sheet, pd.DataFrame)
        assert "annualized_return" in tear_sheet.index
        assert "maximum_drawdown" in tear_sheet.index
        assert "sharpe_ratio" in tear_sheet.index

        # 3. Weights history matches assets and sums to ~1.0 on rebalance dates
        weights_df = results["weights"].dropna()
        assert list(weights_df.columns) == sample_assets
        for _, row in weights_df.iterrows():
            assert np.isclose(row.sum(), 1.0, atol=1e-5)

        # 4. Portfolio returns length matches rebalanced test period
        port_returns = results["returns"]
        assert len(port_returns) > 0
        assert not port_returns.isna().any()
