import numpy as np
import pandas as pd
import pytest
from pyalloq_backtest.costs import (
    AlmgrenChrissCostModel,
    CompositeCostModel,
    FlatBpsCostModel,
    ProportionalSpreadCostModel,
)


class TestCostModels:
    def test_flat_bps_cost_model(self) -> None:
        model = FlatBpsCostModel(bps=10.0)  # 10 bps = 0.0010
        weights_delta = pd.Series([0.2, -0.3, 0.1])
        costs = model.calculate_costs(weights_delta)

        expected = pd.Series([0.2 * 0.001, 0.3 * 0.001, 0.1 * 0.001])
        np.testing.assert_allclose(costs.values, expected.values, rtol=1e-6)

    def test_proportional_spread_cost_model_scalar_and_dict(self) -> None:
        # Scalar spread = 10 bps (half-spread = 5 bps) + 2 bps commission = 7 bps
        scalar_model = ProportionalSpreadCostModel(spread_bps=10.0, commission_bps=2.0)
        delta = pd.Series({"AAPL": 0.2, "MSFT": -0.4})
        costs = scalar_model.calculate_costs(delta)
        np.testing.assert_allclose(
            costs.values, [0.2 * 0.0007, 0.4 * 0.0007], rtol=1e-6
        )

        # Asset-specific spread dictionary
        dict_model = ProportionalSpreadCostModel(
            spread_bps={"AAPL": 4.0, "MSFT": 20.0}, commission_bps=1.0
        )
        dict_costs = dict_model.calculate_costs(delta)
        # AAPL: 0.5*4 + 1 = 3 bps; MSFT: 0.5*20 + 1 = 11 bps
        np.testing.assert_allclose(
            dict_costs.values, [0.2 * 0.0003, 0.4 * 0.0011], rtol=1e-6
        )

        # DataFrame support
        df_delta = pd.DataFrame({"AAPL": [0.1, -0.2], "MSFT": [0.3, 0.0]})
        df_costs = dict_model.calculate_costs(df_delta)
        assert isinstance(df_costs, pd.DataFrame)
        assert np.isclose(df_costs.loc[0, "AAPL"], 0.1 * 0.0003)
        assert np.isclose(df_costs.loc[0, "MSFT"], 0.3 * 0.0011)

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

    def test_composite_cost_model(self) -> None:
        with pytest.raises(ValueError, match="at least one cost model"):
            CompositeCostModel([])

        m1 = FlatBpsCostModel(bps=5.0)
        m2 = ProportionalSpreadCostModel(spread_bps=10.0)  # half-spread = 5 bps
        combo = CompositeCostModel([m1, m2])

        delta = pd.Series({"AAPL": 0.5, "MSFT": -0.5})
        total = combo.calculate_costs(delta)
        np.testing.assert_allclose(
            total.values, [0.5 * 0.0010, 0.5 * 0.0010], rtol=1e-6
        )
