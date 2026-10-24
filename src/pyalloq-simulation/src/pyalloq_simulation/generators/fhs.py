import numpy as np
import pandas as pd
from typing import Any, Literal, cast
import warnings
from arch import arch_model

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class FHSGenerator(BaseScenarioGenerator):
    """
    Filtered Historical Simulation (FHS) Monte Carlo Engine.

    Fits univariate GARCH(1,1) models to strip volatility clustering,
    resamples standardized residuals synchronously to preserve empirical
    correlation, and re-injects dynamic volatility forward.
    """

    def __init__(
        self,
        dist: str = "Normal",
        dtype: Any = np.float32,
        impute_missing: bool = True,
    ) -> None:
        if dist not in ("Normal", "t", "skewt"):
            raise KeyError(f"dist: {dist} needs to be 'Normal', 't' or 'skewt'")
        self.dist = dist
        self.dtype = np.dtype(dtype)
        self.impute_missing = impute_missing
        self.garch_params: dict[str, Any] = {}
        self.last_volatility: dict[str, Any] = {}
        self.log_returns: bool = False
        self.is_fitted = False

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        """
        De-Volatizes the historical time series to extract empirical shocks.
        """
        self.data: MarketData = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        self.log_returns = log_returns
        self._record_index_metadata(self.data.prices.index)
        self.initial_prices = self.data.prices.iloc[-1].to_numpy(dtype=self.dtype)
        self.std_residuals = pd.DataFrame(
            index=self.data.prices.index, columns=self.data.assets, dtype=np.float64
        )
        self.N = len(self.data.assets)
        returns = self.data.get_returns(log_returns)
        self._fit_exogenous_features(self.data, returns)

        # Suppress arch convergence warnings for cleaner SDK output
        warnings.filterwarnings("ignore", category=FutureWarning)
        scale = 100.0
        arch_dist = cast(
            Literal["normal", "t", "skewt"],
            "normal" if self.dist == "Normal" else self.dist,
        )
        for asset in self.data.assets:
            scaled_ret = returns[asset].dropna() * scale
            am = arch_model(
                scaled_ret,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist=arch_dist,
                rescale=False,
            )

            res = am.fit(disp="off")
            params = res.params.copy()
            params["mu"] = params["mu"] / scale
            params["omega"] = params["omega"] / (scale**2)
            self.garch_params[asset] = params

            # z_t = (r_t - mu) / sigma_t (scale cancels out in ratio)
            cond_vol = pd.Series(res.conditional_volatility, index=scaled_ret.index)
            resid_s = pd.Series(res.resid, index=scaled_ret.index)
            self.std_residuals[asset] = resid_s / cond_vol
            # Store the final historical volatility state (unscaled) to seed t=0 of the simulation
            self.last_volatility[asset] = float(cond_vol.iloc[-1]) / scale

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

        mu = np.array(
            [self.garch_params[a]["mu"] for a in self.data.assets], dtype=self.dtype
        )
        omega = np.array(
            [self.garch_params[a]["omega"] for a in self.data.assets], dtype=self.dtype
        )
        alpha = np.array(
            [self.garch_params[a]["alpha[1]"] for a in self.data.assets],
            dtype=self.dtype,
        )
        beta = np.array(
            [self.garch_params[a]["beta[1]"] for a in self.data.assets],
            dtype=self.dtype,
        )

        Z_hist = self.std_residuals.to_numpy(dtype=self.dtype)
        current_vol = np.tile(
            np.array(
                [self.last_volatility[a] for a in self.data.assets], dtype=self.dtype
            ),
            (n_paths, 1),
        )
        synthetic_returns = np.zeros((n_paths, horizon, self.N), dtype=self.dtype)
        random_indices = np.random.randint(0, T_hist, size=(n_paths, horizon))

        for t in range(horizon):
            idx = random_indices[:, t]
            Z_t = Z_hist[idx, :]
            sim_returns = mu + (current_vol * Z_t)
            synthetic_returns[:, t, :] = sim_returns
            current_var = (
                omega + alpha * (current_vol * Z_t) ** 2 + beta * (current_vol**2)
            )
            current_vol = np.sqrt(np.maximum(current_var, 1e-12)).astype(
                self.dtype, copy=False
            )

        if self.log_returns:
            synthetic_prices = self.initial_prices * np.exp(
                np.cumsum(synthetic_returns, axis=1)
            )
        else:
            synthetic_prices = self.initial_prices * np.cumprod(
                1.0 + synthetic_returns, axis=1
            )

        synthetic_prices = synthetic_prices.astype(self.dtype, copy=False)
        sim_features, feat_cols = self._simulate_exogenous_features(synthetic_returns)

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=synthetic_returns,
            asset_names=list(self.data.assets),
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )
