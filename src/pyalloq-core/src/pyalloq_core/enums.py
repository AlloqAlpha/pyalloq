from enum import Enum, auto


class ObjectiveFunction(Enum):
    MAX_SHARPE = auto()
    MIN_VOLATILITY = auto()
    MAX_RETURN = auto()
    RISK_PARITY = auto()
    EFFICIENT_RETURN = auto()
    EFFICIENT_RISK = auto()
    MIN_CVAR = auto()
    MIN_CDAR = auto()
    WASSERSTEIN_KELLY = auto()
    WASSERSTEIN_CVAR = auto()
    WASSERSTEIN_MEAN_RISK = auto()


class ConstraintType(Enum):
    LONG_ONLY = auto()
    MARKET_NEUTRAL = auto()
    CARDINALITY = auto()  # Max number of assets


class DataFrequency(Enum):
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    CRYPTO_DAILY = 365


class CovariateKey(str, Enum):
    """
    Standardized canonical keys for exogenous time-series covariates stored in
    `MarketData.features` (past-only) and `MarketData.future_features` (known-future).

    Inherits from `str` so enum members work directly as dictionary keys in
    `MarketData(features={CovariateKey.COT_CROWDING: df, ...})`.
    """

    IMPLIED_VOL = "implied_vol"
    IMPLIED_SKEW = "implied_skew"
    IMPLIED_KURTOSIS = "implied_kurtosis"
    INDEX_IMPLIED_VOL = "index_implied_vol"
    INTEREST_RATES = "interest_rates"
    COT_CROWDING = "cot_crowding"
    MACRO_INDICATORS = "macro_indicators"
    VIX = "vix"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    CALENDAR_EVENTS = "calendar_events"
    EARNINGS_SCHEDULE = "earnings_schedule"


class CovariateMode(Enum):
    """
    Controls how multivariate foundation models (`ApolloPFN`, `TimesFM-3`, `Timestral`)
    extract optional exogenous covariates from `MarketData`.
    """

    AUTO = auto()
    NONE = auto()
    ASSET_ONLY = auto()
    MACRO_ONLY = auto()
    SELECTED = auto()
