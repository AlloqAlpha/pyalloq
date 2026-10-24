from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Standardized temporal output of a strategy running across a time horizon."""

    name: str
    returns: pd.Series
    weights: pd.DataFrame
    tear_sheet: pd.DataFrame

    _ACCESSIBLE_KEYS: ClassVar[tuple[str, ...]] = (
        "name",
        "returns",
        "weights",
        "tear_sheet",
        "equity_curve",
    )

    @property
    def equity_curve(self) -> pd.Series:
        """Dynamically computes the equity curve from daily returns."""
        return (1.0 + self.returns).cumprod()

    def __getitem__(self, key: str) -> Any:
        if key in self._ACCESSIBLE_KEYS:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._ACCESSIBLE_KEYS

    def __iter__(self):
        return iter(self._ACCESSIBLE_KEYS)

    def __len__(self) -> int:
        return len(self._ACCESSIBLE_KEYS)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self._ACCESSIBLE_KEYS else default

    def keys(self) -> list[str]:
        return list(self._ACCESSIBLE_KEYS)

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self._ACCESSIBLE_KEYS]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self._ACCESSIBLE_KEYS]


@dataclass
class OptimizationResult:
    """Standardized output for all portfolio optimizers."""

    name = str
    weights: pd.Series
    status: str  # e.g., "OPTIMAL", "INFEASIBLE", "SUBOPTIMAL"

    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Store solver-specific data (e.g., cvxpy objective value, RL episode reward, HRP linkages)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clean_weights(self, cutoff: float = 1e-4) -> pd.Series:
        clean = self.weights.copy()
        clean[np.abs(clean) < cutoff] = 0.0

        return clean / clean.sum()


@dataclass
class MonteCarloResult:
    """Standardized output for a multi-path synthetic backtest."""

    returns_matrix: pd.DataFrame
    equity_curves: pd.DataFrame
    metrics_distribution: pd.DataFrame
