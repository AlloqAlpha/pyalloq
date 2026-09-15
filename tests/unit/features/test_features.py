import numpy as np
from pyalloq_core.data import MarketData
from pyalloq_features.core.pipeline import FeaturePipeline
from pyalloq_features.scalers.rolling import RollingZScoreScaler
from pyalloq_features.transformers.technical import TechnicalIndicator


class TestFeaturePipeline:
    def test_rolling_zscore_scaler(self, synthetic_market_data: MarketData) -> None:
        window = 30
        scaler = RollingZScoreScaler(target_feature="volume", window=window)
        out_data = scaler.transform(synthetic_market_data)

        out_col = f"volume_zscore_{window}"
        assert out_col in out_data.features
        z_df = out_data.features[out_col]

        assert z_df.shape == synthetic_market_data.prices.shape
        # Rolling z-scores after burn-in period should have approximately mean 0
        valid_values = z_df.iloc[window:].values.flatten()
        assert not np.isnan(valid_values).all()
        assert abs(np.nanmean(valid_values)) < 0.5

    def test_technical_indicator_sma(self, synthetic_market_data: MarketData) -> None:
        sma_tf = TechnicalIndicator(indicator="SMA", source="prices", timeperiod=10)
        out_data = sma_tf.transform(synthetic_market_data)

        assert "SMA_10" in out_data.features
        sma_df = out_data.features["SMA_10"]
        assert sma_df.shape == synthetic_market_data.prices.shape
        # Check first 9 values are NaN for period 10
        assert np.isnan(sma_df.iloc[0:9].values).all()
        # Check that 10th value is equal to simple average of first 10 prices
        first_10_mean = synthetic_market_data.prices.iloc[0:10].mean()
        np.testing.assert_allclose(
            sma_df.iloc[9].values, first_10_mean.values, rtol=1e-5
        )

    def test_feature_pipeline_chaining(self, synthetic_market_data: MarketData) -> None:
        pipeline = FeaturePipeline()
        pipeline.add(
            TechnicalIndicator(indicator="RSI", source="prices", timeperiod=14)
        )
        pipeline.add(RollingZScoreScaler(target_feature="volume", window=20))

        result = pipeline.run(synthetic_market_data)
        assert "RSI_14" in result.features
        assert "volume_zscore_20" in result.features

    def test_cross_sectional_rank(self, synthetic_market_data: MarketData) -> None:
        from pyalloq_features.transformers.cross_sectional import CrossSectionalRank

        cs_rank = CrossSectionalRank(target_feature="volume")
        out_data = cs_rank.transform(synthetic_market_data)

        assert "volume_cs_rank" in out_data.features
        ranks = out_data.features["volume_cs_rank"]
        # In percentile rank across 4 assets, ranks must be between 0 and 1
        assert (ranks >= 0.0).all().all()
        assert (ranks <= 1.0).all().all()
