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

        # Constant returns have 0 standard deviation, hence maximum drawdown & ulcer index are 0
        np.testing.assert_allclose(metrics["maximum_drawdown"], 0.0, atol=1e-6)
        np.testing.assert_allclose(metrics["ulcer_index"], 0.0, atol=1e-6)
        assert metrics["omega_ratio"] == np.inf

    def test_maximum_drawdown_negative(self) -> None:
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        # Starts at 100, drops to 80 (20% drawdown) then recovers
        returns = pd.Series([0.0, -0.1, -0.111111, 0.1, 0.1], index=dates)
        sheet = MetricsTearSheet.generate(returns)
        mdd = sheet.loc["maximum_drawdown", "Value"]
        ulcer = sheet.loc["ulcer_index", "Value"]

        assert mdd <= 0.0, "Maximum drawdown must be non-positive"
        assert np.isclose(mdd, -0.2, atol=1e-3)
        assert ulcer > 0.0

    def test_extended_institutional_metrics_and_benchmark(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-02", periods=252, freq="B")
        bench_ret = pd.Series(rng.normal(0.0005, 0.01, size=252), index=dates)
        port_ret = 1.2 * bench_ret + 0.0005 + rng.normal(0.0, 0.003, size=252)

        weights = pd.DataFrame(
            {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, index=dates
        )
        turnover = pd.Series(0.01, index=dates)  # 1% daily -> 252% annualized
        rf_series = pd.Series(0.02, index=dates)

        sheet = MetricsTearSheet.generate(
            port_ret,
            risk_free_rate=rf_series,
            weights=weights,
            turnover=turnover,
            benchmark_returns=bench_ret,
        )
        m = sheet["Value"].to_dict()

        assert 0.0 <= m["probabilistic_sharpe_ratio"] <= 1.0
        assert m["omega_ratio"] > 1.0
        assert m["tail_ratio"] > 0.0
        assert np.isclose(m["annualized_turnover"], 2.52, rtol=1e-5)
        assert np.isclose(m["effective_number_of_bets"], 4.0, rtol=1e-5)
        assert m["tracking_error"] > 0.0
        assert "information_ratio" in m
        assert np.isclose(m["beta"], 1.2, atol=0.15)
        assert "alpha" in m

    def test_empty_returns_handling(self) -> None:
        empty = pd.Series(dtype=float)
        sheet = MetricsTearSheet.generate(empty)
        assert sheet.loc["total_return", "Value"] == 0.0
        assert sheet.loc["sharpe_ratio", "Value"] == 0.0
