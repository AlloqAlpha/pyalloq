import numpy as np
import pandas as pd
import pytest
from pyalloq_backtest.costs import AlmgrenChrissCostModel, FlatBpsCostModel


class TestCostModels:
    def test_flat_bps_cost_model(self) -> None:
        model = FlatBpsCostModel(bps=10.0)  # 10 bps = 0.0010
        weights_delta = pd.Series([0.2, -0.3, 0.1])
        costs = model.calculate_costs(weights_delta)

        expected = pd.Series([0.2 * 0.001, 0.3 * 0.001, 0.1 * 0.001])
        np.testing.assert_allclose(costs.values, expected.values, rtol=1e-6)

    def test_almgren_chriss_requires_volume(self) -> None:
        model = AlmgrenChrissCostModel(portfolio_aum=1e6)
        weights_delta = pd.Series([0.1, -0.1])
        with pytest.raises(ValueError, match="requires 'volume'"):
            model.calculate_costs(weights_delta, features_slice=None)

    def test_almgren_chriss_market_impact(self) -> None:
        model = AlmgrenChrissCostModel(portfolio_aum=1e6, spread_bps=5.0, gamma=0.1)
        weights_delta = pd.Series([0.1], index=["AAPL"])
        features = {"volume": pd.Series([1e7], index=["AAPL"])}

        costs = model.calculate_costs(weights_delta, features_slice=features)
        assert isinstance(costs, pd.Series)
        assert costs["AAPL"] > 0.0
        # Larger trade dollar size should increase percentage cost due to market impact
        large_delta = pd.Series([0.5], index=["AAPL"])
        large_costs = model.calculate_costs(large_delta, features_slice=features)
        # Unit percentage cost for larger trade must be strictly higher
        unit_cost_small = costs["AAPL"] / weights_delta["AAPL"]
        unit_cost_large = large_costs["AAPL"] / large_delta["AAPL"]
        assert unit_cost_large > unit_cost_small
