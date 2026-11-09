from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseCovarianceEstimator
from sklearn.covariance import LedoitWolf


class ImpliedCovarianceEstimator(BaseCovarianceEstimator):
    """
    Buss & Vilkov (2012) Point-in-Time Option-Implied Covariance Estimator.

    Combines forward-looking asset implied volatilities (from `data.features[implied_vol_feature]`,
    such as 30-day Model-Free Implied Volatility produced by `PITOptionsMFIVTransformer`)
    with a shrunk physical correlation matrix and the Buss-Vilkov implied correlation adjustment:

        Gamma_imp = (1 - gamma_t) * Gamma_hist + gamma_t * (1 @ 1^T)
        Sigma_imp = D_IV @ Gamma_imp @ D_IV

    When an index/market implied volatility `index_implied_vol` is provided (via constructor,
    `estimate(..., index_implied_vol=...)`, or `data.features["index_implied_vol"]`), `gamma_t`
    is solved in closed form so that the implied portfolio variance under market weights `w_M`
    matches `(index_implied_vol)^2`:

        gamma_t = (sigma_{M,IV}^2 - w_M^T D_IV Gamma_hist D_IV w_M)
                  / ((w_M^T sigma_IV)^2 - w_M^T D_IV Gamma_hist D_IV w_M)

    When `index_implied_vol` is not supplied, `gamma_t` defaults to a regime-adaptive
    variance-risk-premium weight comparing current implied volatility to historical realized
    volatility (plus `correlation_risk_premium`).
    """

    def __init__(
        self,
        implied_vol_feature: str = "implied_vol",
        correlation_window: int = 60,
        annualization_factor: float = 252.0,
        index_implied_vol: float | None = None,
        market_weights: pd.Series | dict[str, float] | None = None,
        correlation_risk_premium: float = 0.05,
        iv_is_annualized: bool = True,
        use_cornish_fisher_tail: bool = False,
        tail_confidence: float = 0.95,
    ) -> None:
        self.implied_vol_feature = implied_vol_feature
        self.correlation_window = int(correlation_window)
        self.annualization_factor = float(annualization_factor)
        self.index_implied_vol = index_implied_vol
        self.market_weights = market_weights
        self.correlation_risk_premium = float(correlation_risk_premium)
        self.iv_is_annualized = bool(iv_is_annualized)
        self.use_cornish_fisher_tail = bool(use_cornish_fisher_tail)
        self.tail_confidence = float(np.clip(tail_confidence, 0.55, 0.999))
        self.last_implied_correlation_shift_: float = 0.0
        self.last_cornish_fisher_multiplier_: np.ndarray | None = None

    @staticmethod
    def compute_cornish_fisher_multiplier(
        skew: np.ndarray,
        excess_kurtosis: np.ndarray,
        confidence: float = 0.95,
    ) -> np.ndarray:
        """
        Computes the Cornish-Fisher tail volatility multiplier for each asset given
        return skewness `S_i` (where negative return skew `S_i < 0` increases downside tail risk)
        and excess kurtosis `K_i`:
            m_CF,i = 1 - ((z^2 - 1) / (6 z)) * S_i + ((z^3 - 3 z) / (24 z)) * K_i
                     - ((2 z^3 - 5 z) / (36 z)) * S_i^2
        """
        from scipy.stats import norm

        z = float(norm.ppf(np.clip(confidence, 0.55, 0.999)))
        s = np.clip(np.asarray(skew, dtype=np.float64), -3.0, 3.0)
        k = np.clip(np.asarray(excess_kurtosis, dtype=np.float64), -1.5, 10.0)
        mult = (
            1.0
            - ((z**2 - 1.0) / (6.0 * z)) * s
            + ((z**2 - 3.0) / 24.0) * k
            - ((2.0 * (z**2) - 5.0) / 36.0) * (s**2)
        )
        return np.clip(mult, 0.65, 2.50)

    def _extract_point_in_time_iv(
        self, data: MarketData, returns: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the latest Point-in-Time annualized asset implied volatility vector
        sigma_IV in R^N alongside trailing annualized realized volatility sigma_RV in R^N.
        """
        tickers = list(data.prices.columns)
        window_ret = returns.iloc[-self.correlation_window :]
        rv_series = window_ret.std(ddof=1).fillna(0.01) * np.sqrt(
            self.annualization_factor
        )
        rv_vec = np.clip(rv_series.to_numpy(dtype=np.float64), 1e-4, 5.0)

        if self.implied_vol_feature in data.features:
            iv_df = data.features[self.implied_vol_feature]
            latest_iv = iv_df.iloc[-1].reindex(tickers).to_numpy(dtype=np.float64)
            if not self.iv_is_annualized:
                latest_iv = latest_iv * np.sqrt(self.annualization_factor)
            valid_mask = np.isfinite(latest_iv) & (latest_iv > 0.0)
            iv_vec = np.where(valid_mask, latest_iv, rv_vec)
        else:
            ewm_std = (
                returns.ewm(span=min(20, max(2, len(returns))), min_periods=2)
                .std()
                .iloc[-1]
                .reindex(tickers)
                .fillna(0.01)
                .to_numpy(dtype=np.float64)
            ) * np.sqrt(self.annualization_factor)
            iv_vec = np.clip(ewm_std, 1e-4, 5.0)

        return np.clip(iv_vec, 1e-4, 5.0), rv_vec

    def _compute_shrunk_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Computes a positive-definite Ledoit-Wolf shrunk correlation matrix over `correlation_window`.
        """
        n_assets = returns.shape[1]
        if n_assets == 1:
            return np.ones((1, 1), dtype=np.float64)

        win = returns.iloc[-self.correlation_window :].dropna(how="all").fillna(0.0)
        if len(win) < 3:
            return np.eye(n_assets, dtype=np.float64)

        raw = win.to_numpy(dtype=np.float64)
        std = np.std(raw, axis=0, ddof=1)
        std = np.where(std > 1e-10, std, 1.0)
        standardized = (raw - np.mean(raw, axis=0)) / std

        try:
            lw = LedoitWolf().fit(standardized)
            cov_z = lw.covariance_.astype(np.float64)
        except Exception:
            cov_z = np.cov(standardized, rowvar=False)

        d_inv = 1.0 / np.sqrt(np.clip(np.diag(cov_z), 1e-10, None))
        corr = cov_z * np.outer(d_inv, d_inv)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _resolve_market_weights(self, data: MarketData) -> np.ndarray:
        tickers = list(data.prices.columns)
        n_assets = len(tickers)
        if self.market_weights is not None:
            w_ser = pd.Series(self.market_weights).reindex(tickers).fillna(0.0)
            w = w_ser.to_numpy(dtype=np.float64)
            if w.sum() > 0:
                return w / w.sum()

        if (
            data.cross_sectional is not None
            and "market_cap" in data.cross_sectional.columns
        ):
            mc = (
                data.cross_sectional["market_cap"]
                .reindex(tickers)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            if mc.sum() > 0:
                return mc / mc.sum()

        return np.full(n_assets, 1.0 / max(1, n_assets), dtype=np.float64)

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.DataFrame:
        tickers = list(data.prices.columns)
        n_assets = len(tickers)
        returns = data.prices.pct_change().dropna(how="all").fillna(0.0)

        iv_vec, rv_vec = self._extract_point_in_time_iv(data, returns)
        if self.use_cornish_fisher_tail:
            if "implied_skew" in data.features:
                skew_vec = (
                    data.features["implied_skew"]
                    .iloc[-1]
                    .reindex(tickers)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                skew_vec = (
                    returns.iloc[-self.correlation_window :]
                    .skew()
                    .reindex(tickers)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            if "implied_kurtosis" in data.features:
                kurt_vec = (
                    data.features["implied_kurtosis"]
                    .iloc[-1]
                    .reindex(tickers)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                kurt_vec = (
                    returns.iloc[-self.correlation_window :]
                    .kurt()
                    .reindex(tickers)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
            cf_mult = self.compute_cornish_fisher_multiplier(
                skew_vec, kurt_vec, confidence=self.tail_confidence
            )
            self.last_cornish_fisher_multiplier_ = cf_mult
            iv_vec = np.clip(iv_vec * cf_mult, 1e-4, 5.0)
        corr_hist = self._compute_shrunk_correlation(returns)
        w_m = self._resolve_market_weights(data)

        idx_iv = kwargs.get("index_implied_vol", self.index_implied_vol)
        if idx_iv is None and "index_implied_vol" in data.features:
            idx_df = data.features["index_implied_vol"]
            val = float(idx_df.to_numpy(dtype=np.float64)[-1, 0])
            if np.isfinite(val) and val > 0:
                idx_iv = (
                    val
                    if self.iv_is_annualized
                    else val * float(np.sqrt(self.annualization_factor))
                )

        weighted_iv = w_m * iv_vec
        var_hist_corr = float(weighted_iv @ corr_hist @ weighted_iv)
        var_perfect_corr = float((np.sum(weighted_iv)) ** 2)
        denom = var_perfect_corr - var_hist_corr

        if idx_iv is not None and denom > 1e-10:
            target_idx_var = float(idx_iv) ** 2
            gamma_raw = (target_idx_var - var_hist_corr) / denom
        else:
            avg_iv_rv_ratio = float(np.mean(iv_vec / np.clip(rv_vec, 1e-4, None)))
            stress_shift = np.tanh(max(0.0, avg_iv_rv_ratio - 1.0)) * 0.35
            gamma_raw = self.correlation_risk_premium + stress_shift

        eigvals_hist = np.linalg.eigvalsh(corr_hist)
        min_eig = float(max(eigvals_hist[0], 1e-6))
        lower_bound = -min_eig / max(n_assets - min_eig, 1.0) + 1e-4
        gamma_t = float(np.clip(gamma_raw, lower_bound, 0.95))
        self.last_implied_correlation_shift_ = gamma_t

        ones_mat = np.ones((n_assets, n_assets), dtype=np.float64)
        corr_imp = (1.0 - gamma_t) * corr_hist + gamma_t * ones_mat
        np.fill_diagonal(corr_imp, 1.0)

        cov_imp = np.outer(iv_vec, iv_vec) * corr_imp
        cov_imp = 0.5 * (cov_imp + cov_imp.T)

        return pd.DataFrame(cov_imp, index=tickers, columns=tickers)
