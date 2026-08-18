from enum import Enum, auto


class ObjectiveFunction(Enum):
    MAX_SHARPE = auto()
    MIN_VOLATILITY = auto()
    MAX_RETURN = auto()
    RISK_PARITY = auto()


class ConstraintType(Enum):
    LONG_ONLY = auto()
    MARKET_NEUTRAL = auto()
    CARDINALITY = auto()  # Max number of assets


class DataFrequency(Enum):
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    CRYPTO_DAILY = 365
