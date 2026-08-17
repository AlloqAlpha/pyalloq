from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


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
