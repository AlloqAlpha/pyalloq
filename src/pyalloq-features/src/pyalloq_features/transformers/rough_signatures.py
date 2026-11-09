from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn
from pyalloq_core.data import MarketData
from pyalloq_features.core.base import BaseFeatureTransformer


class LogSignatureTransformer(BaseFeatureTransformer):
    """
    Truncated Log-Signature Transformer in the Lyndon Basis (Depth 2 & Depth 3).

    Compresses local windows of multi-channel asset trajectories
    X_t^{(i)} = (t_norm, log_P_t^{(i)}, sigma_t^{(i)}) into geometric path tokens
    via the Baker-Campbell-Hausdorff (BCH) expansion:
    - Level 1: Total increments Delta X^a_{s, t}
    - Level 2: Lévy Areas (Lie brackets [e_a, e_b] = 1/2 int (X^a_u - X^a_s) dX^b_u - (X^b_u - X^b_s) dX^a_u)
    - Level 3: Nested Lie brackets [e_a, [e_a, e_b]] capturing path curvature and roughness asymmetry.

    Populates:
    - `data.features["logsig_levy_price_vol"]`: Price-volatility Lévy area (signed lead-lag between returns & vol)
    - `data.features["logsig_time_price_area"]`: Time-price Lévy area (within-window momentum convexity)
    - `data.features["logsig_norm"]`: Carnot-Carathéodory / Euclidean norm of the local log-signature vector.
    """

    def __init__(
        self,
        window: int = 10,
        vol_feature: str = "implied_vol",
        depth: int = 2,
    ) -> None:
        if window < 3:
            raise ValueError(
                "`window` must be at least 3 to compute iterated integrals."
            )
        if depth not in (2, 3):
            raise ValueError("`depth` must be 2 or 3.")
        self.window = int(window)
        self.vol_feature = vol_feature
        self.depth = int(depth)

    @staticmethod
    def compute_window_log_signature(
        path: np.ndarray,
        depth: int = 2,
    ) -> np.ndarray:
        """
        Computes the truncated Log-Signature in the Lyndon basis for a single multivariate
        window `path` of shape (W, D) using piecewise-linear Chen's identity / BCH formula.
        """
        W, D = path.shape
        if W < 2:
            n_l2 = (D * (D - 1)) // 2
            return np.zeros(D + n_l2, dtype=np.float64)

        # Center path at start of window X_0 = 0
        rel_path = path - path[0:1, :]
        dX = np.diff(rel_path, axis=0)  # (W - 1, D)
        X_mid = 0.5 * (
            rel_path[:-1, :] + rel_path[1:, :]
        )  # Trapezoidal midpoint (W - 1, D)

        # Level 1: Total displacement S^1_a = X_T^a - X_0^a
        level_1 = rel_path[-1, :]  # (D,)

        # Level 2: Lévy area A_{a, b} = 1/2 sum_{k} (X_{mid, k}^a dX_k^b - X_{mid, k}^b dX_k^a) for a < b
        # By Baker-Campbell-Hausdorff, the Lyndon bracket coefficient [e_a, e_b] equals the Lévy area.
        levy_pairs: list[float] = []
        level_3_brackets: list[float] = []

        for a in range(D):
            for b in range(a + 1, D):
                area_ab = 0.5 * float(
                    np.sum(X_mid[:, a] * dX[:, b] - X_mid[:, b] * dX[:, a])
                )
                levy_pairs.append(area_ab)

                if depth == 3:
                    # Running Lévy area along the window for nested Lie bracket [e_a, [e_a, e_b]]
                    dA_ab = 0.5 * (X_mid[:, a] * dX[:, b] - X_mid[:, b] * dX[:, a])
                    running_area = np.cumsum(dA_ab)
                    b_aab = (1.0 / 6.0) * float(
                        np.sum(X_mid[:, a] * dA_ab - running_area * dX[:, a])
                    )
                    level_3_brackets.append(b_aab)

        if depth == 3:
            return np.concatenate(
                [
                    level_1,
                    np.asarray(levy_pairs, dtype=np.float64),
                    np.asarray(level_3_brackets, dtype=np.float64),
                ]
            )
        return np.concatenate([level_1, np.asarray(levy_pairs, dtype=np.float64)])

    def transform(self, data: MarketData) -> MarketData:
        prices = data.prices
        n_obs, n_assets = prices.shape
        tickers = list(prices.columns)

        log_prices = np.log(prices.clip(lower=1e-8).to_numpy(dtype=np.float64))
        returns = prices.pct_change().fillna(0.0)

        if self.vol_feature in data.features:
            vol_mat = (
                data.features[self.vol_feature]
                .reindex(index=prices.index, columns=tickers)
                .ffill()
                .bfill()
                .fillna(0.15)
                .to_numpy(dtype=np.float64)
            )
        else:
            vol_mat = returns.ewm(span=self.window, min_periods=2).std().bfill().fillna(
                0.01
            ).to_numpy(dtype=np.float64) * np.sqrt(252.0)

        levy_pv = np.zeros((n_obs, n_assets), dtype=np.float64)
        levy_tp = np.zeros((n_obs, n_assets), dtype=np.float64)
        logsig_norm = np.zeros((n_obs, n_assets), dtype=np.float64)

        W = self.window
        for j in range(n_assets):
            lp_j = log_prices[:, j]
            v_j = vol_mat[:, j]
            for t in range(2, n_obs):
                start_idx = max(0, t - W + 1)
                win_lp = lp_j[start_idx : t + 1]
                win_v = v_j[start_idx : t + 1]
                t_sub = np.linspace(0.0, 1.0, len(win_lp), dtype=np.float64)
                path = np.column_stack([t_sub, win_lp, win_v])
                lsig = self.compute_window_log_signature(path, depth=self.depth)
                # Channels: 0=t, 1=log_p, 2=vol -> Pairs: (0,1)=t-p, (0,2)=t-v, (1,2)=p-v
                levy_tp[t, j] = lsig[3]
                levy_pv[t, j] = lsig[5]
                logsig_norm[t, j] = float(np.linalg.norm(lsig))

        data.features["logsig_levy_price_vol"] = pd.DataFrame(
            levy_pv, index=prices.index, columns=tickers
        )
        data.features["logsig_time_price_area"] = pd.DataFrame(
            levy_tp, index=prices.index, columns=tickers
        )
        data.features["logsig_norm"] = pd.DataFrame(
            logsig_norm, index=prices.index, columns=tickers
        )
        return data


class LiftedHestonRoughVolTransformer(BaseFeatureTransformer):
    """
    Markovian Lifted Heston Rough Volatility & Adaptive Hurst Transformer (Abi Jaber 2019).

    Resolves both computational intractability of non-Markovian fractional kernels and the
    Microstructure Noise Critique (Cont & Das 2023):
    1. Estimates both the backward-looking realized Hurst exponent H_RV in (0.05, 0.50]
       and the forward-looking implied Hurst exponent H_IV in (0.20, 0.50] (when
       `data.features["implied_vol"]` is available), blending them via a microstructure
       noise gate into an effective Hurst exponent `H_eff`.
    2. Lifts the rough fractional power-law kernel K(t) = t^{H - 1/2} / Gamma(H + 1/2)
       into a finite superposition of K Markovian mean-reverting factors U_{t, k}
       with geometric mean-reversion speeds x_k = r^{k - K/2} and analytic weights c_k(H):
           V_t = g_0(t) + sum_{k=1}^K c_k(H_eff) U_{t, k}

    Populates:
    - `data.features["lifted_heston_vol"]`: Multi-factor Markovian rough volatility forecast
    - `data.features["hurst_exponent"]`: Point-in-time effective Hurst parameter H_eff in (0.05, 0.50].
    """

    def __init__(
        self,
        n_factors: int = 5,
        speed_ratio: float = 2.5,
        estimation_window: int = 30,
        implied_vol_feature: str = "implied_vol",
        annualization_factor: float = 252.0,
        vol_of_vol: float = 0.30,
    ) -> None:
        self.n_factors = int(max(2, n_factors))
        self.speed_ratio = float(max(1.2, speed_ratio))
        self.estimation_window = int(max(10, estimation_window))
        self.implied_vol_feature = implied_vol_feature
        self.annualization_factor = float(annualization_factor)
        self.vol_of_vol = float(vol_of_vol)

    @staticmethod
    def lifted_heston_weights_and_speeds(
        hurst: float,
        n_factors: int = 5,
        r_ratio: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes theAbi Jaber (2019) geometric mean-reversion speeds x_k and weights c_k
        approximating the fractional kernel K(t) = t^{H - 1/2} / Gamma(H + 1/2).
        """
        H = float(np.clip(hurst, 0.05, 0.49))
        alpha = H + 0.5  # in (0.55, 0.99)
        idx = np.arange(1, n_factors + 1, dtype=np.float64) - 0.5 * (n_factors + 1)
        speeds = np.power(r_ratio, idx)

        prefactor = (np.power(r_ratio, 1.0 - alpha) - 1.0) / (
            gamma_fn(alpha) * gamma_fn(2.0 - alpha)
        )
        weights = prefactor * np.power(speeds, 1.0 - alpha)
        # Normalize weights so total factor responsiveness is well-scaled across H
        w_sum = float(np.sum(weights))
        if w_sum > 1e-12:
            weights = weights / w_sum
        return weights, speeds

    @staticmethod
    def estimate_hurst_variogram(log_vol_series: np.ndarray) -> float:
        """
        Estimates the Hurst exponent H in (0.05, 0.50] from the second-order variogram
        E[|log(sigma_{t+lag}) - log(sigma_t)|^2] ~ C * lag^{2 H} across lags {1, 2, 4, 8}.
        """
        clean = log_vol_series[np.isfinite(log_vol_series)]
        if len(clean) < 10:
            return 0.25

        lags = [lag for lag in (1, 2, 4, 8) if lag < len(clean) // 2]
        if len(lags) < 2:
            return 0.25

        log_lags: list[float] = []
        log_vars: list[float] = []
        for lag in lags:
            diffs = clean[lag:] - clean[:-lag]
            m2 = float(np.mean(diffs**2))
            if m2 > 1e-12:
                log_lags.append(float(np.log(lag)))
                log_vars.append(float(np.log(m2)))

        if len(log_lags) < 2:
            return 0.25

        slope, _ = np.polyfit(log_lags, log_vars, 1)
        hurst = 0.5 * float(slope)
        return float(np.clip(hurst, 0.05, 0.50))

    def transform(self, data: MarketData) -> MarketData:
        prices = data.prices
        n_obs, n_assets = prices.shape
        tickers = list(prices.columns)
        returns = prices.pct_change().fillna(0.0)

        # Short-horizon realized volatility proxy (annualized)
        rv_df = (
            returns.ewm(span=5, min_periods=2).std().bfill().fillna(0.01)
            * np.sqrt(self.annualization_factor)
        ).clip(lower=0.02, upper=3.0)
        rv_mat = rv_df.to_numpy(dtype=np.float64)

        has_iv = self.implied_vol_feature in data.features
        if has_iv:
            iv_mat = (
                data.features[self.implied_vol_feature]
                .reindex(index=prices.index, columns=tickers)
                .ffill()
                .bfill()
                .fillna(rv_df)
                .to_numpy(dtype=np.float64)
            )
        else:
            iv_mat = rv_mat.copy()

        lifted_vol_out = np.zeros((n_obs, n_assets), dtype=np.float64)
        hurst_out = np.full((n_obs, n_assets), 0.25, dtype=np.float64)
        dt = 1.0 / self.annualization_factor
        ret_mat = returns.to_numpy(dtype=np.float64)

        for j in range(n_assets):
            log_rv = np.log(np.clip(rv_mat[:, j], 1e-4, None))
            log_iv = np.log(np.clip(iv_mat[:, j], 1e-4, None))

            # Multi-scale Markovian factor state U_{t, k} in R^K
            u_factors = np.zeros(self.n_factors, dtype=np.float64)
            base_var = float(iv_mat[0, j] ** 2)

            for t in range(n_obs):
                if t >= self.estimation_window:
                    h_rv = self.estimate_hurst_variogram(
                        log_rv[t - self.estimation_window : t + 1]
                    )
                    if has_iv:
                        h_iv = self.estimate_hurst_variogram(
                            log_iv[t - self.estimation_window : t + 1]
                        )
                        # Microstructure noise gate: blend realized roughness with implied surface Hurst
                        h_eff = 0.45 * h_rv + 0.55 * h_iv
                    else:
                        h_eff = h_rv
                else:
                    h_eff = 0.25

                hurst_out[t, j] = h_eff
                weights, speeds = self.lifted_heston_weights_and_speeds(
                    hurst=h_eff,
                    n_factors=self.n_factors,
                    r_ratio=self.speed_ratio,
                )

                # Slow forward variance baseline g_0(t) anchored by PIT implied vol or expanding RV
                g0_t = 0.7 * float(iv_mat[t, j] ** 2) + 0.3 * base_var
                base_var = 0.96 * base_var + 0.04 * float(rv_mat[t, j] ** 2)

                # Standardized return squared shock driving the K Markovian factors
                inst_var = max(1e-6, g0_t + float(np.dot(weights, u_factors)))
                shock = (ret_mat[t, j] ** 2) * self.annualization_factor - inst_var

                decay = np.exp(-speeds * dt)
                u_factors = decay * u_factors + self.vol_of_vol * shock * (1.0 - decay)

                v_lifted = max(1e-4, g0_t + float(np.dot(weights, u_factors)))
                lifted_vol_out[t, j] = float(np.clip(np.sqrt(v_lifted), 0.02, 3.0))

        data.features["lifted_heston_vol"] = pd.DataFrame(
            lifted_vol_out, index=prices.index, columns=tickers
        )
        data.features["hurst_exponent"] = pd.DataFrame(
            hurst_out, index=prices.index, columns=tickers
        )
        return data
