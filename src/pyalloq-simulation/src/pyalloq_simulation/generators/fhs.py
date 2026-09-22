import numpy as np
import pandas as pd
from typing import Any
import warnings
from arch import arch_model  # type: ignore[import-not-found]

from pyalloq_core.interfaces import BaseScenarioGenerator, ScenarioMarketData
from pyalloq_core.data import MarketData


class FHSGenerator(BaseScenarioGenerator):
    """
    Filtered Historical Simulation (FHS) Monte Carlo Engine.

    Fits univariate GARCH(1,1) models to strip volatility clustering,
    resamples standardized residuals synchronously to preserve empirical
    correlation, and re-injects dynamic volatility forward.
    """

    def __init__(self, dist: str = "Normal") -> None:
        if dist not in ("Normal", "t", "skewt"):
            raise KeyError(f"dist: {dist} needs to be 'Normal', 't' or 'skewt'")
        self.dist = dist
        self.garch_params: dict[str, Any] = {}
        self.last_volatility: dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, data: MarketData, log_returns: bool = False):
        """
        De-volatizes the historical time series to extract empirical shocks.
        """
        self.data = data
        self.std_residuals = pd.DataFrame(
            index=self.data.prices.index, columns=self.data.assets
        )
        self.N = len(self.data.assets)
        returns = data.get_returns(log_returns)
        # Suppress arch convergence warnings for cleaner SDK output
        warnings.filterwarnings("ignore", category=FutureWarning)
        for asset in self.data.assets:
            am = arch_model(
                returns[asset].dropna(),
                mean="Constant",
                vol="Garch",
                p=1,
                q=1,
                dist=self.dist,
            )

            res = am.fit(disp="off")
            self.garch_params[asset] = res.params
            # z_t = (r_t - mu) / sigma_t
            self.std_residuals[asset] = res.resid / res.conditional_volatility
            # Store the final historical volatility state to seed t=0 of the simulation
            self.last_volatility[asset] = res.conditional_volatility.iloc[-1]

        # Drop rows with any NaNs to ensure a perfectly synchronous residual matrix
        self.std_residuals.dropna(inplace=True)
        self.is_fitted = True

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        """
        Generates forward-looking synthetic paths.
        Args:
            n_paths (int): No. of Monte Carlo scenarios (M).
            horizon (int): Forward time steps to simulate (T).

        Returns:
            ScenarioMarketData: A dataclass wrapping the (M, T, N) tensor.
        """

        if not self.is_fitted:
            raise ValueError(
                "FHS needs to be fitted. Call .fit(data: MarketData) first"
            )

        T_hist = len(self.std_residuals)

        mu = np.array([self.garch_params[a]["mu"] for a in self.data.assets])
        omega = np.array([self.garch_params[a]["omega"] for a in self.data.assets])
        alpha = np.array([self.garch_params[a]["alpha[1]"] for a in self.data.assets])
        beta = np.array([self.garch_params[a]["beta[1]"] for a in self.data.assets])

        Z_hist = self.std_residuals.to_numpy()
        current_vol = np.tile(
            [self.last_volatility[a] for a in self.data.assets], (n_paths, 1)
        )
        synthetic_returns = np.zeros((n_paths, horizon, self.N))
        random_indices = np.random.randint(0, T_hist, size=(n_paths, horizon))

        for t in range(horizon):
            idx = random_indices[:, t]
            Z_t = Z_hist[idx, :]
            sim_returns = mu + (current_vol * Z_t)
            synthetic_returns[:, t, :] = sim_returns
            current_var = (
                omega + alpha * (current_vol * Z_t) ** 2 + beta * (current_vol**2)
            )
            current_vol = np.sqrt(current_var)

        initial_prices = self.data.prices.iloc[-1].to_numpy(dtype=float)
        synthetic_prices = initial_prices * np.cumprod(1 + synthetic_returns, axis=1)

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=synthetic_returns,
            asset_names=list(self.data.assets),
        )
