import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.max_diversification import MaxDiversificationAllocator
from pyalloq.classical.optimizers.naive import EqualWeightAllocator, RandomAllocator
from pyalloq.classical.optimizers.risk_budgeting import RiskBudgetingAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator


class TestClassicalOptimizers:
    def test_equal_weight_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        allocator = EqualWeightAllocator(tickers=sample_assets)
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        res = allocator.allocate(synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0)
        for asset in sample_assets:
            assert np.isclose(res.weights[asset], 1.0 / len(sample_assets))

    def test_random_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        allocator = RandomAllocator(tickers=sample_assets, seed=42)
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        res = allocator.allocate(synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0)
        assert np.all(res.weights.values >= 0.0)

    def test_markowitz_min_volatility_analytical_solution(self) -> None:
        # Two-asset analytical benchmark
        # Sigma = [[0.04, 0.0], [0.0, 0.09]]
        # Theoretical optimal weight w1 = sigma2^2 / (sigma1^2 + sigma2^2) = 0.09 / 0.13 = 9/13
        tickers = ["ASSET_1", "ASSET_2"]
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.09]],
            index=tickers,
            columns=tickers,
        )
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = MarketData(
            prices=pd.DataFrame(100.0, index=dates, columns=tickers), risk_free_rate=0.0
        )

        allocator = MarkowitzAllocator(
            tickers=tickers, objective=ObjectiveFunction.MIN_VOLATILITY
        )
        res = allocator.allocate(data=data, cov_matrix=cov)

        w = res.weights
        assert np.isclose(w.sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(w["ASSET_1"], 9.0 / 13.0, atol=1e-4)
        np.testing.assert_allclose(w["ASSET_2"], 4.0 / 13.0, atol=1e-4)

    def test_markowitz_min_volatility_variance_reduction(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        allocator = MarkowitzAllocator(
            tickers=sample_assets, objective=ObjectiveFunction.MIN_VOLATILITY
        )
        res = allocator.allocate(synthetic_market_data, cov_matrix=cov)

        w_opt = res.weights.values
        var_opt = float(w_opt.T @ cov.values @ w_opt)

        w_eq = np.ones(len(sample_assets)) / len(sample_assets)
        var_eq = float(w_eq.T @ cov.values @ w_eq)

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert np.all(res.weights.values >= -1e-6)
        assert (
            var_opt <= var_eq + 1e-6
        ), "Min volatility portfolio must have variance <= equal weight portfolio"

    def test_markowitz_max_sharpe(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        mu = pd.Series([0.15, 0.12, 0.10, 0.08], index=sample_assets)
        allocator = MarkowitzAllocator(
            tickers=sample_assets, objective=ObjectiveFunction.MAX_SHARPE
        )
        res = allocator.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert np.all(res.weights.values >= -1e-6)

    def test_risk_parity_equal_risk_contributions(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        allocator = RiskParityAllocator(tickers=sample_assets)
        res = allocator.allocate(data=synthetic_market_data, cov_matrix=cov)

        w = res.weights.values
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert np.all(w >= 0.0)

        # Mathematical Invariant: Total Risk Contribution TRC_i = w_i * (Sigma w)_i / sigma_p
        sigma_p = np.sqrt(w.T @ cov.values @ w)
        mrc = (cov.values @ w) / sigma_p
        trc = w * mrc

        # For equal risk parity, every asset's TRC must be equal: TRC_i = sigma_p / N
        expected_trc = sigma_p / len(sample_assets)
        np.testing.assert_allclose(
            trc, expected_trc, rtol=1e-2, err_msg="TRC must be equal across assets"
        )

    def test_max_diversification_ratio(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        allocator = MaxDiversificationAllocator(tickers=sample_assets)
        res = allocator.allocate(synthetic_market_data, cov_matrix=cov)

        w = res.weights.values
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert np.all(w >= -1e-6)

        vols = np.sqrt(np.diag(cov.values))
        sigma_p = np.sqrt(w.T @ cov.values @ w)
        dr_opt = (w @ vols) / sigma_p

        w_eq = np.ones(len(sample_assets)) / len(sample_assets)
        sigma_p_eq = np.sqrt(w_eq.T @ cov.values @ w_eq)
        dr_eq = (w_eq @ vols) / sigma_p_eq

        assert (
            dr_opt >= dr_eq - 1e-4
        ), "Max diversification ratio must be >= equal weight diversification ratio"

    def test_risk_budgeting_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        budgets = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_assets)
        synthetic_market_data.features["risk_budgets"] = budgets

        allocator = RiskBudgetingAllocator(tickers=sample_assets)
        res = allocator.allocate(data=synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert np.all(res.weights.values >= 0.0)
