import numpy as np
import pandas as pd
from pyalloq_backtest.metrics import MetricsTearSheet


class TestMetrics:
    def test_metrics_analytical_benchmarks(self) -> None:
        # Constant positive daily returns of 0.001 (10 bps/day) over 252 days
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        daily_return = 0.001
        portfolio_returns = pd.Series(daily_return, index=dates)

        sheet = MetricsTearSheet.generate(portfolio_returns, risk_free_rate=0.0)
        metrics = sheet["Value"].to_dict()

        # Analytical expectations
        expected_total_return = (1 + daily_return) ** 252 - 1
        np.testing.assert_allclose(
            metrics["total_return"], expected_total_return, rtol=1e-5
        )
        np.testing.assert_allclose(
            metrics["annualized_return"], expected_total_return, rtol=1e-5
        )

        # Constant returns have 0 standard deviation, hence maximum drawdown is 0
        np.testing.assert_allclose(metrics["maximum_drawdown"], 0.0, atol=1e-6)

    def test_maximum_drawdown_negative(self) -> None:
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        # Starts at 100, drops to 80 (20% drawdown) then recovers
        returns = pd.Series([0.0, -0.1, -0.111111, 0.1, 0.1], index=dates)
        sheet = MetricsTearSheet.generate(returns)
        mdd = sheet.loc["maximum_drawdown", "Value"]

        assert mdd <= 0.0, "Maximum drawdown must be non-positive"
        assert np.isclose(mdd, -0.2, atol=1e-3)
