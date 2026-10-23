import numpy as np
from sklearn.decomposition import PCA

from pyalloq_core.data import MarketData


class FactorBackcastImputer:
    """
    Imputes missing asset return and price histories (e.g., recent IPOs, internal gaps,
    or delisted assets) using PCA Factor Back-Casting in return space.

    Preserves the full historical length T of the portfolio universe instead of
    truncating to the shortest-lived asset via .dropna().
    """

    def __init__(
        self,
        n_components: int = 3,
        min_coverage: float = 0.85,
        inject_idiosyncratic_noise: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self.n_components = n_components
        self.min_coverage = min_coverage
        self.inject_idiosyncratic_noise = inject_idiosyncratic_noise
        self.random_state = random_state

    def fit_transform(self, data: MarketData) -> MarketData:
        prices = data.prices.copy()
        if not prices.isna().any().any():
            return data

        rng = np.random.default_rng(self.random_state)

        # Compute raw percentage returns without dropping rows across columns
        raw_returns = prices.pct_change(fill_method=None).iloc[1:]
        T, N = raw_returns.shape
        if T == 0 or N == 0:
            return data

        # Identify anchor assets with sufficient historical coverage
        coverage = raw_returns.notna().mean(axis=0)
        anchor_cols = coverage[coverage >= self.min_coverage].index.tolist()
        if not anchor_cols:
            # Fallback to the asset(s) with highest coverage
            max_cov = coverage.max()
            anchor_cols = coverage[coverage >= max_cov].index.tolist()

        anchor_returns = raw_returns[anchor_cols].fillna(0.0).to_numpy(dtype=np.float64)
        k_eff = min(self.n_components, len(anchor_cols), max(1, T - 1))

        if k_eff >= 1 and len(anchor_cols) > 1:
            pca = PCA(n_components=k_eff, random_state=self.random_state)
            factors = pca.fit_transform(anchor_returns)  # Shape: (T, k_eff)
        else:
            factors = np.nanmean(
                raw_returns.to_numpy(dtype=np.float64), axis=1, keepdims=True
            )
            factors = np.nan_to_num(factors, nan=0.0)
            k_eff = 1

        # Design matrix with intercept: [1, F] of shape (T, k_eff + 1)
        X_full = np.column_stack([np.ones(T, dtype=np.float64), factors])
        reg_eye = 1e-5 * np.eye(k_eff + 1, dtype=np.float64)
        reg_eye[0, 0] = 0.0

        imputed_returns = raw_returns.copy()

        for col in raw_returns.columns:
            series = raw_returns[col]
            missing_mask = series.isna().to_numpy()
            if not np.any(missing_mask):
                continue

            obs_mask = ~missing_mask
            n_obs = int(np.sum(obs_mask))

            if n_obs >= k_eff + 2:
                X_obs = X_full[obs_mask]
                y_obs = series.to_numpy(dtype=np.float64)[obs_mask]
                beta = np.linalg.solve(X_obs.T @ X_obs + reg_eye, X_obs.T @ y_obs)
                y_hat_obs = X_obs @ beta
                resid_std = float(np.std(y_obs - y_hat_obs))

                X_miss = X_full[missing_mask]
                y_miss = X_miss @ beta
                if self.inject_idiosyncratic_noise and resid_std > 0:
                    y_miss = y_miss + rng.normal(0.0, resid_std, size=len(y_miss))
            elif n_obs > 0:
                obs_vals = series.to_numpy(dtype=np.float64)[obs_mask]
                mu = float(np.mean(obs_vals))
                sigma = float(np.std(obs_vals)) if n_obs > 1 else 0.01
                y_miss = rng.normal(mu, sigma, size=int(np.sum(missing_mask)))
            else:
                y_miss = np.zeros(int(np.sum(missing_mask)), dtype=np.float64)

            # Clamp extreme synthetic single-period returns below -90% to avoid negative/exploding back-cast prices
            y_miss = np.clip(y_miss, -0.90, 5.0)
            col_vals = series.to_numpy(dtype=np.float64).copy()
            col_vals[missing_mask] = y_miss
            imputed_returns[col] = col_vals

        # Reconstruct continuous price paths anchored on observed prices
        reconstructed_prices = prices.copy()
        ret_matrix = imputed_returns.to_numpy(dtype=np.float64)

        for j, col in enumerate(prices.columns):
            p_col = prices[col].to_numpy(dtype=np.float64).copy()
            if not np.any(np.isnan(p_col)):
                continue

            valid_indices = np.where(~np.isnan(p_col))[0]
            if len(valid_indices) == 0:
                reconstructed_prices[col] = 100.0 * np.cumprod(
                    np.concatenate([[1.0], 1.0 + ret_matrix[:, j]])
                )
                continue

            first_valid = int(valid_indices[0])
            # Back-cast prior to first valid price (e.g. pre-IPO history)
            for t in range(first_valid - 1, -1, -1):
                r_next = ret_matrix[
                    t, j
                ]  # ret_matrix[t] corresponds to (P_{t+1}/P_t - 1)
                denom = max(1.0 + r_next, 0.1)
                p_col[t] = p_col[t + 1] / denom

            # Forward-cast any interior or trailing NaNs after first_valid
            for t in range(first_valid + 1, len(p_col)):
                if np.isnan(p_col[t]):
                    r_curr = ret_matrix[t - 1, j]
                    p_col[t] = p_col[t - 1] * max(1.0 + r_curr, 0.1)

            reconstructed_prices[col] = p_col

        return MarketData(
            assets=list(data.assets),
            prices=reconstructed_prices,
            features={k: v.ffill().bfill() for k, v in data.features.items()},
            future_features=data.future_features,
            cross_sectional=data.cross_sectional,
            risk_free_rate=data.risk_free_rate,
            risk_aversion=data.risk_aversion,
        )

    def transform(self, data: MarketData) -> MarketData:
        return self.fit_transform(data)
