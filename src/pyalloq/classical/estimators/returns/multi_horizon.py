from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator

from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator


class AlphaDecayReturnEstimator(BaseReturnEstimator):
    """
    Multi-horizon return estimator modeling exponential alpha decay (Gârleanu & Pedersen, 2013).
    Projects a base single-period return forecast across H future stages:
        mu_{t + tau} = (1 - phi)^tau * mu_t + (1 - (1 - phi)^tau) * mu_equilibrium
    """

    def __init__(
        self,
        base_estimator: BaseReturnEstimator | None = None,
        horizon: int = 5,
        alpha_decay: float = 0.15,
        equilibrium_mode: str = "zero",
    ) -> None:
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        if not (0.0 <= alpha_decay <= 1.0):
            raise ValueError("`alpha_decay` (phi) must be in [0.0, 1.0].")
        self.base_estimator = base_estimator or EWMAReturnEstimator(span=60)
        self.horizon = horizon
        self.alpha_decay = alpha_decay
        self.equilibrium_mode = equilibrium_mode

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        """Returns the immediate 1-step expected return Series (BaseReturnEstimator contract)."""
        return self.base_estimator.estimate(data, **kwargs)

    def estimate_horizon(
        self,
        data: MarketData,
        horizon: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Returns a 2D DataFrame of shape (H, N) representing forecasted returns
        for stages tau = 0, 1, ..., H - 1.
        """
        h_steps = horizon if horizon is not None else self.horizon
        mu_0 = self.base_estimator.estimate(data, **kwargs)
        assets = list(mu_0.index)
        mu_0_vals = mu_0.to_numpy(dtype=np.float64)

        if self.equilibrium_mode == "grand_mean":
            mu_eq = np.full_like(mu_0_vals, float(np.mean(mu_0_vals)))
        elif self.equilibrium_mode == "historical_mean":
            rets = data.prices.reindex(columns=assets).pct_change().dropna()
            mu_eq = (
                rets.mean().to_numpy(dtype=np.float64) * 252.0
                if len(rets) > 0
                else np.zeros_like(mu_0_vals)
            )
        else:
            mu_eq = np.zeros_like(mu_0_vals)

        rows = []
        persistence = 1.0 - self.alpha_decay
        for tau in range(h_steps):
            decay_factor = persistence**tau
            mu_tau = decay_factor * mu_0_vals + (1.0 - decay_factor) * mu_eq
            rows.append(mu_tau)

        stage_index = pd.Index([f"t+{tau + 1}" for tau in range(h_steps)], name="Stage")
        return pd.DataFrame(rows, index=stage_index, columns=assets)

    def estimate_trajectory(
        self,
        data: MarketData,
        horizon: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Alias for `estimate_horizon` returning the (H, N) multi-stage forecast DataFrame."""
        return self.estimate_horizon(data, horizon=horizon, **kwargs)
