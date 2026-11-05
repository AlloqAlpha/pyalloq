import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.optimizers.cvar import CDaRAllocator, MeanCVaRAllocator
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator
from pyalloq.classical.optimizers.max_diversification import MaxDiversificationAllocator
from pyalloq.classical.optimizers.naive import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    MarketCapWeightAllocator,
    RandomAllocator,
)
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

    def test_inverse_volatility_and_market_cap_allocators(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        inv_vol = InverseVolatilityAllocator(tickers=sample_assets)
        res_iv = inv_vol.allocate(synthetic_market_data, cov_matrix=cov)
        assert np.isclose(res_iv.weights.sum(), 1.0)
        assert np.all(res_iv.weights.values > 0.0)

        mcaps = pd.DataFrame(
            {"market_cap": [400.0, 300.0, 200.0, 100.0]}, index=sample_assets
        )
        synthetic_market_data.cross_sectional = mcaps
        mcap_alloc = MarketCapWeightAllocator(tickers=sample_assets)
        res_mc = mcap_alloc.allocate(synthetic_market_data, cov_matrix=cov)
        np.testing.assert_allclose(
            res_mc.weights.values, np.array([0.4, 0.3, 0.2, 0.1]), atol=1e-6
        )

    def test_markowitz_constraints_and_efficient_frontier(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        mu = pd.Series([0.18, 0.14, 0.10, 0.06], index=sample_assets)

        # 1. Box bounds + Sector constraints + L2 regularization + Turnover constraint
        sector_map = {
            sample_assets[0]: "Tech",
            sample_assets[1]: "Tech",
            sample_assets[2]: "Finance",
            sample_assets[3]: "Finance",
        }
        w_prev = pd.Series(0.25, index=sample_assets)
        alloc_constrained = MarkowitzAllocator(
            tickers=sample_assets,
            objective=ObjectiveFunction.MAX_SHARPE,
            weight_bounds=(0.05, 0.50),
            l2_gamma=0.05,
            max_turnover=0.60,
            sector_mapper=sector_map,
            sector_upper={"Tech": 0.65},
            sector_lower={"Finance": 0.35},
        )
        res_c = alloc_constrained.allocate(
            synthetic_market_data,
            cov_matrix=cov,
            expected_returns=mu,
            current_weights=w_prev,
        )
        assert np.isclose(res_c.weights.sum(), 1.0, atol=1e-4)
        assert np.all(res_c.weights.values >= 0.05 - 1e-3)
        assert np.all(res_c.weights.values <= 0.50 + 1e-3)
        assert (
            res_c.weights[sample_assets[0]] + res_c.weights[sample_assets[1]]
            <= 0.65 + 1e-3
        )
        assert np.sum(np.abs(res_c.weights - w_prev)) <= 0.60 + 1e-3

        # 2. Dollar-neutral / Market-neutral portfolio
        alloc_mn = MarkowitzAllocator(
            tickers=sample_assets,
            objective=ObjectiveFunction.MAX_RETURN,
            weight_bounds=(-0.5, 0.5),
            market_neutral=True,
            max_leverage=1.0,
        )
        res_mn = alloc_mn.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )
        assert np.isclose(res_mn.weights.sum(), 0.0, atol=1e-4)
        assert np.sum(np.abs(res_mn.weights.values)) <= 1.0 + 1e-4

        # 3. Efficient Return & Efficient Risk & Frontier
        alloc_er = MarkowitzAllocator(
            tickers=sample_assets,
            objective=ObjectiveFunction.EFFICIENT_RETURN,
            target_return=0.12,
        )
        res_er = alloc_er.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )
        assert (
            res_er.expected_return is not None and res_er.expected_return >= 0.12 - 1e-3
        )

        alloc_risk = MarkowitzAllocator(
            tickers=sample_assets,
            objective=ObjectiveFunction.EFFICIENT_RISK,
            target_volatility=0.25,
        )
        res_risk = alloc_risk.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )
        assert res_risk.volatility is not None and res_risk.volatility <= 0.25 + 1e-3

        frontier = alloc_er.efficient_frontier(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu, n_points=5
        )
        assert len(frontier) >= 3
        vols = [pt.volatility for pt in frontier if pt.volatility is not None]
        assert vols == sorted(vols, key=lambda v: round(v, 4))

    def test_mean_cvar_and_cdar_allocators(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        mu = pd.Series([0.15, 0.12, 0.10, 0.08], index=sample_assets)

        cvar_alloc = MeanCVaRAllocator(
            tickers=sample_assets,
            beta=0.95,
            objective=ObjectiveFunction.MIN_CVAR,
            weight_bounds=(0.05, 0.60),
        )
        res_cvar = cvar_alloc.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )
        assert res_cvar.status == "OPTIMAL_CVAR"
        assert np.isclose(res_cvar.weights.sum(), 1.0, atol=1e-4)
        assert np.all(res_cvar.weights.values >= 0.05 - 1e-4)
        assert "daily_cvar" in res_cvar.metadata

        cdar_alloc = CDaRAllocator(
            tickers=sample_assets,
            beta=0.90,
            weight_bounds=(0.0, 0.70),
        )
        res_cdar = cdar_alloc.allocate(
            synthetic_market_data, cov_matrix=cov, expected_returns=mu
        )
        assert res_cdar.status == "OPTIMAL_CDAR"
        assert np.isclose(res_cdar.weights.sum(), 1.0, atol=1e-4)
        assert "cdar" in res_cdar.metadata
