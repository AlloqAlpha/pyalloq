import numpy as np
from pyalloq_core.data import MarketData

from pyalloq.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.optimizers.machine_learning.herc import HERCAllocator
from pyalloq.optimizers.machine_learning.hrp import HRPAllocator
from pyalloq.optimizers.machine_learning.nco import NCOAllocator


class TestMachineLearningOptimizers:
    def test_hrp_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        hrp = HRPAllocator(tickers=sample_assets)
        res = hrp.allocate(synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert np.all(res.weights.values >= 0.0)
        assert list(res.weights.index) == sample_assets

    def test_hrp_collinear_asset_stability(
        self, collinear_market_data: MarketData
    ) -> None:
        cov = EmpiricalCovariance().estimate(collinear_market_data)
        tickers = list(collinear_market_data.prices.columns)
        hrp = HRPAllocator(tickers=tickers)
        res = hrp.allocate(collinear_market_data, cov_matrix=cov)

        # HRP should never produce negative or NaN weights, even with perfectly collinear assets
        assert not res.weights.isna().any()
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert np.all(res.weights.values >= 0.0)

    def test_herc_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        herc = HERCAllocator(tickers=sample_assets)
        res = herc.allocate(synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert np.all(res.weights.values >= 0.0)
        assert list(res.weights.index) == sample_assets

    def test_nco_allocator(
        self, synthetic_market_data: MarketData, sample_assets: list[str]
    ) -> None:
        cov = EmpiricalCovariance().estimate(synthetic_market_data)
        # 4 assets partitioned into 2 clusters
        nco = NCOAllocator(tickers=sample_assets, n_clusters=2)
        res = nco.allocate(synthetic_market_data, cov_matrix=cov)

        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert np.all(res.weights.values >= -1e-6)
        assert list(res.weights.index) == sample_assets
