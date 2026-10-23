from typing import Any
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_simulation.utils.imputation import FactorBackcastImputer


class GARCHEVTCopulaGenerator(BaseScenarioGenerator):
    """
    GARCH-EVT-Student-t Copula Monte Carlo Generator.

    1. Filters univariate marginals via AR(1)-GARCH(1,1) to extract i.i.d. standardized residuals.
    2. Models marginal tails with Extreme Value Theory (Peaks-Over-Threshold Generalized Pareto
       Distribution) and the interior with the empirical distribution.
    3. Binds marginals via a Student-t Copula with correlation matrix R and degrees of freedom nu
       to capture asymmetric tail risk and lower-tail dependence (lambda_L > 0).
    """

    def __init__(
        self,
        nu: float = 5.0,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
        mean_model: str = "AR",
        dtype: Any = np.float32,
        impute_missing: bool = True,
    ) -> None:
        if nu <= 2.0:
            raise ValueError("Student-t degrees of freedom `nu` must be > 2.0.")
        if not (0.01 <= lower_quantile < upper_quantile <= 0.99):
            raise ValueError(
                "Quantile thresholds must satisfy 0.01 <= lower_quantile < upper_quantile <= 0.99."
            )
        self.nu = float(nu)
        self.q_lower = float(lower_quantile)
        self.q_upper = float(upper_quantile)
        self.mean_model = mean_model
        self.dtype = np.dtype(dtype)
        self.impute_missing = impute_missing

        self.is_fitted = False
        self.log_returns: bool = False
        self.assets: list[str] = []
        self.N: int = 0
        self.garch_params: dict[str, dict[str, float]] = {}
        self.evt_params: dict[str, dict[str, Any]] = {}
        self.copula_chol: np.ndarray | None = None
        self.initial_prices: np.ndarray | None = None

    def _fit_evt_marginal(self, z: np.ndarray) -> dict[str, Any]:
        z_clean = np.sort(z[~np.isnan(z)])
        u_l = float(np.quantile(z_clean, self.q_lower))
        u_u = float(np.quantile(z_clean, self.q_upper))

        lower_exceed = u_l - z_clean[z_clean < u_l]
        if len(lower_exceed) >= 5:
            xi_l, _, scale_l = stats.genpareto.fit(lower_exceed, floc=0.0)
            xi_l = float(np.clip(xi_l, -0.4, 0.45))
            scale_l = float(max(scale_l, 1e-4))
        else:
            xi_l, scale_l = 0.1, float(max(np.std(z_clean) * 0.5, 0.1))

        upper_exceed = z_clean[z_clean > u_u] - u_u
        if len(upper_exceed) >= 5:
            xi_u, _, scale_u = stats.genpareto.fit(upper_exceed, floc=0.0)
            xi_u = float(np.clip(xi_u, -0.4, 0.45))
            scale_u = float(max(scale_u, 1e-4))
        else:
            xi_u, scale_u = 0.1, float(max(np.std(z_clean) * 0.5, 0.1))

        return {
            "sorted_z": z_clean,
            "u_l": u_l,
            "u_u": u_u,
            "xi_l": xi_l,
            "scale_l": scale_l,
            "xi_u": xi_u,
            "scale_u": scale_u,
        }

    def _evt_cdf(self, z: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        u = np.empty_like(z, dtype=np.float64)
        u_l = meta["u_l"]
        u_u = meta["u_u"]
        sorted_z = meta["sorted_z"]
        n = len(sorted_z)

        mask_l = z < u_l
        mask_u = z > u_u
        mask_mid = ~(mask_l | mask_u)

        if np.any(mask_l):
            exceed_l = u_l - z[mask_l]
            gpd_cdf_l = stats.genpareto.cdf(
                exceed_l, c=meta["xi_l"], loc=0.0, scale=meta["scale_l"]
            )
            u[mask_l] = self.q_lower * (1.0 - gpd_cdf_l)

        if np.any(mask_u):
            exceed_u = z[mask_u] - u_u
            gpd_cdf_u = stats.genpareto.cdf(
                exceed_u, c=meta["xi_u"], loc=0.0, scale=meta["scale_u"]
            )
            u[mask_u] = self.q_upper + (1.0 - self.q_upper) * gpd_cdf_u

        if np.any(mask_mid):
            ranks = np.searchsorted(sorted_z, z[mask_mid], side="right")
            u[mask_mid] = np.clip(ranks / (n + 1.0), self.q_lower, self.q_upper)

        return np.clip(u, 1e-5, 1.0 - 1e-5)

    def _evt_ppf(self, u: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        u_clipped = np.clip(u, 1e-5, 1.0 - 1e-5)
        z_out = np.empty_like(u_clipped, dtype=np.float64)
        u_l = meta["u_l"]
        u_u = meta["u_u"]
        sorted_z = meta["sorted_z"]

        mask_l = u_clipped < self.q_lower
        mask_u = u_clipped > self.q_upper
        mask_mid = ~(mask_l | mask_u)

        if np.any(mask_l):
            tail_prob_l = np.clip(
                1.0 - (u_clipped[mask_l] / self.q_lower), 1e-6, 1.0 - 1e-6
            )
            exceed_l = stats.genpareto.ppf(
                tail_prob_l, c=meta["xi_l"], loc=0.0, scale=meta["scale_l"]
            )
            z_out[mask_l] = u_l - exceed_l

        if np.any(mask_u):
            tail_prob_u = np.clip(
                (u_clipped[mask_u] - self.q_upper) / (1.0 - self.q_upper),
                1e-6,
                1.0 - 1e-6,
            )
            exceed_u = stats.genpareto.ppf(
                tail_prob_u, c=meta["xi_u"], loc=0.0, scale=meta["scale_u"]
            )
            z_out[mask_u] = u_u + exceed_u

        if np.any(mask_mid):
            interior_z = sorted_z[(sorted_z >= u_l) & (sorted_z <= u_u)]
            if len(interior_z) >= 2:
                rel_q = (u_clipped[mask_mid] - self.q_lower) / (
                    self.q_upper - self.q_lower
                )
                z_out[mask_mid] = np.quantile(interior_z, np.clip(rel_q, 0.0, 1.0))
            else:
                z_out[mask_mid] = np.quantile(sorted_z, u_clipped[mask_mid])

        # Standardize and bound extreme tail draws for multi-step GARCH stability
        return np.clip(z_out, -12.0, 12.0)

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        clean_data = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        self.assets = list(clean_data.assets)
        self.N = len(self.assets)
        self.log_returns = log_returns
        self._record_index_metadata(clean_data.prices.index)
        self.initial_prices = clean_data.prices.iloc[-1].to_numpy(dtype=self.dtype)

        returns = clean_data.get_returns(log_returns)
        self._fit_exogenous_features(clean_data, returns)

        warnings.filterwarnings("ignore", category=FutureWarning)
        scale = 100.0
        std_residuals_df = pd.DataFrame(
            index=returns.index, columns=self.assets, dtype=np.float64
        )

        use_ar = self.mean_model.upper() == "AR" and len(returns) > 30
        for asset in self.assets:
            series = returns[asset].dropna() * scale
            am = arch_model(
                series,
                mean="AR" if use_ar else "Constant",
                lags=1 if use_ar else 0,
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
                rescale=False,
            )
            res = am.fit(disp="off")
            params = res.params

            mu_val = float(params.get("Const", params.get("mu", 0.0))) / scale
            ar1_val = float(params.get(f"{asset}[1]", 0.0)) if use_ar else 0.0
            # Fallback lookup if arch named the AR lag differently
            if use_ar and ar1_val == 0.0:
                for k_param in params.index:
                    if "[1]" in str(k_param) and k_param not in ("alpha[1]", "beta[1]"):
                        ar1_val = float(params[k_param])
                        break

            omega_val = float(params["omega"]) / (scale**2)
            alpha_val = float(params["alpha[1]"])
            beta_val = float(params["beta[1]"])

            cond_vol = pd.Series(res.conditional_volatility, index=series.index)
            resid_s = pd.Series(res.resid, index=series.index)
            z_series = (resid_s / cond_vol).dropna()
            std_residuals_df.loc[z_series.index, asset] = z_series.to_numpy(
                dtype=np.float64
            )

            self.garch_params[asset] = {
                "mu": mu_val,
                "ar1": float(np.clip(ar1_val, -0.95, 0.95)),
                "omega": omega_val,
                "alpha": alpha_val,
                "beta": beta_val,
                "last_vol": float(cond_vol.iloc[-1]) / scale,
                "last_ret": float(returns[asset].iloc[-1]),
            }
            self.evt_params[asset] = self._fit_evt_marginal(
                z_series.to_numpy(dtype=np.float64)
            )

        std_residuals_df.dropna(inplace=True)
        Z_mat = std_residuals_df.to_numpy(dtype=np.float64)

        # Transform marginals to Uniform via EVT CDF, then to Student-t quantiles
        U_mat = np.zeros_like(Z_mat)
        for j, asset in enumerate(self.assets):
            U_mat[:, j] = self._evt_cdf(Z_mat[:, j], self.evt_params[asset])

        T_mat = stats.t.ppf(U_mat, df=self.nu)
        corr = np.corrcoef(T_mat, rowvar=False)
        if corr.ndim == 0:
            corr = np.ones((1, 1), dtype=np.float64)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # Positive-definite repair via eigenvalue flooring if needed
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-4)
        corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_inv = np.diag(1.0 / np.sqrt(np.diag(corr_pd)))
        corr_pd = d_inv @ corr_pd @ d_inv

        self.copula_chol = np.linalg.cholesky(corr_pd)
        self.is_fitted = True

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.copula_chol is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "GARCHEVTCopulaGenerator needs to be fitted. Call .fit() first."
            )

        # 1. Sample multivariate Student-t Copula across (n_paths, horizon, N)
        Z_norm = np.random.standard_normal(size=(n_paths, horizon, self.N))
        correlated_norm = np.matmul(Z_norm, self.copula_chol.T)
        chi2_draws = np.random.chisquare(df=self.nu, size=(n_paths, horizon, 1))
        t_draws = correlated_norm / np.sqrt(chi2_draws / self.nu)
        U_sim = stats.t.cdf(t_draws, df=self.nu)

        # 2. Invert through semi-parametric EVT-GPD marginal quantiles
        Z_shocks = np.empty((n_paths, horizon, self.N), dtype=self.dtype)
        for j, asset in enumerate(self.assets):
            Z_shocks[:, :, j] = self._evt_ppf(
                U_sim[:, :, j], self.evt_params[asset]
            ).astype(self.dtype)

        # 3. Step AR(1)-GARCH(1,1) dynamics forward
        mu = np.array(
            [self.garch_params[a]["mu"] for a in self.assets], dtype=self.dtype
        )
        ar1 = np.array(
            [self.garch_params[a]["ar1"] for a in self.assets], dtype=self.dtype
        )
        omega = np.array(
            [self.garch_params[a]["omega"] for a in self.assets], dtype=self.dtype
        )
        alpha = np.array(
            [self.garch_params[a]["alpha"] for a in self.assets], dtype=self.dtype
        )
        beta = np.array(
            [self.garch_params[a]["beta"] for a in self.assets], dtype=self.dtype
        )

        current_vol = np.tile(
            np.array(
                [self.garch_params[a]["last_vol"] for a in self.assets],
                dtype=self.dtype,
            ),
            (n_paths, 1),
        )
        prev_ret = np.tile(
            np.array(
                [self.garch_params[a]["last_ret"] for a in self.assets],
                dtype=self.dtype,
            ),
            (n_paths, 1),
        )

        synthetic_returns = np.zeros((n_paths, horizon, self.N), dtype=self.dtype)
        for t in range(horizon):
            z_t = Z_shocks[:, t, :]
            eps_t = current_vol * z_t
            r_t = mu + ar1 * prev_ret + eps_t
            synthetic_returns[:, t, :] = r_t
            prev_ret = r_t
            next_var = omega + alpha * (eps_t**2) + beta * (current_vol**2)
            current_vol = np.sqrt(np.maximum(next_var, 1e-12)).astype(
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
            asset_names=self.assets,
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )
