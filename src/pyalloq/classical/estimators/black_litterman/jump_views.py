from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator, BaseViewGenerator
from pyalloq_features.regimes.jump_models import (
    SparseJumpModelDetector,
    extract_jump_regime_features,
)
from sklearn.ensemble import GradientBoostingClassifier


class AssetSpecificJumpViewGenerator(BaseViewGenerator, BaseReturnEstimator):
    """
    Asset- and Factor-Specific Sparse Jump Model + Gradient-Boosted Next-Period Regime
    Forecast View Generator (`AssetSpecificJumpViewGenerator`, Shu, Yu & Mulvey, 2025,
    *Annals of Operations Research*, arXiv:2406.09578; Shu & Mulvey, 2025, *JPM*,
    arXiv:2410.14841).

    Instead of forcing all assets or style factors into a single macro S&P 500 regime:
    1. Fits a persistent `SparseJumpModelDetector` (default 2 or 3 states) independently
       per asset (or style factor) over rolling return-derived risk and Sortino features.
    2. Trains a gradient-boosted classifier (`xgboost.XGBClassifier` when installed, or
       `sklearn.ensemble.GradientBoostingClassifier`) mapping time-`t` asset features to
       next-period regime probabilities `p_hat_{i, t+1} in Delta^K`.
    3. Computes expected return view `Q_i = sum_k p_hat_{i, t+1, k} * mu_{i, k}` and
       regime-mixture uncertainty variance `Omega_{i, i}` for direct injection into
       `BlackLittermanEstimator` or `WassersteinDROAllocator`.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        jump_penalty: float = 15.0,
        sparsity_kappa: float = 2.0,
        windows: Sequence[int] = (5, 20, 60),
        n_estimators: int = 40,
        max_depth: int = 2,
        learning_rate: float = 0.08,
        annualization_factor: float = 252.0,
        use_xgboost_if_available: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = max(2, int(n_regimes))
        self.jump_penalty = float(max(0.0, jump_penalty))
        self.sparsity_kappa = float(max(1.0, sparsity_kappa))
        self.windows = tuple(int(w) for w in windows)
        self.n_estimators = max(10, int(n_estimators))
        self.max_depth = max(1, int(max_depth))
        self.learning_rate = float(learning_rate)
        self.annualization_factor = float(annualization_factor)
        self.use_xgboost_if_available = bool(use_xgboost_if_available)
        self.random_state = int(random_state)

        self.last_next_regime_probs_: pd.DataFrame | None = None
        self.last_posterior_std_: pd.Series | None = None

    def _build_classifier(self) -> Any:
        if self.use_xgboost_if_available:
            try:
                import xgboost as xgb  # type: ignore[import-not-found]

                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    verbosity=0,
                )
            except (ImportError, ValueError, RuntimeError):
                return GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def _forecast_single_asset(
        self,
        asset_name: str,
        asset_prices: pd.Series,
    ) -> tuple[float, float, np.ndarray]:
        """
        Fits SJM on a single asset's price series, predicts next-step regime probabilities
        `p_hat_{i, t+1}`, and returns `(expected_ann_return, ann_view_variance, next_probs)`.
        """
        single_md = MarketData(prices=asset_prices.to_frame(name=asset_name))
        ret_ser = asset_prices.pct_change().dropna()
        if len(ret_ser) < 12:
            ann_mu = (
                float(ret_ser.mean() * self.annualization_factor)
                if len(ret_ser) > 0
                else 0.0
            )
            ann_var = (
                float(ret_ser.var() * self.annualization_factor)
                if len(ret_ser) > 1
                else 0.04
            )
            return (
                ann_mu,
                max(ann_var, 1e-4),
                np.full(self.n_regimes, 1.0 / self.n_regimes),
            )

        sjm = SparseJumpModelDetector(
            n_regimes=self.n_regimes,
            jump_penalty=self.jump_penalty,
            sparsity_kappa=self.sparsity_kappa,
            windows=self.windows,
            annualization_factor=self.annualization_factor,
            random_state=self.random_state,
        )
        reg_state = sjm.detect(single_md)
        states = (
            reg_state.hard_state.reindex(ret_ser.index)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
        k_eff = reg_state.probs.shape[1]

        # Compute regime-conditional annualized return and variance
        ret_np = ret_ser.to_numpy(dtype=np.float64)
        reg_means = np.zeros(k_eff, dtype=np.float64)
        reg_vars = np.zeros(k_eff, dtype=np.float64)
        global_var = max(
            float(np.var(ret_np, ddof=1)) * self.annualization_factor, 1e-4
        )

        for k in range(k_eff):
            mask_k = states == k
            if np.sum(mask_k) >= 2:
                reg_means[k] = (
                    float(np.mean(ret_np[mask_k])) * self.annualization_factor
                )
                reg_vars[k] = max(
                    float(np.var(ret_np[mask_k], ddof=1)) * self.annualization_factor,
                    1e-5,
                )
            elif np.sum(mask_k) == 1:
                reg_means[k] = float(ret_np[mask_k][0]) * self.annualization_factor
                reg_vars[k] = global_var
            else:
                reg_means[k] = float(np.mean(ret_np)) * self.annualization_factor
                reg_vars[k] = global_var

        # Build feature matrix X_t and target y_t = s_{t+1}
        feat_df = extract_jump_regime_features(
            ret_ser.to_frame(name=asset_name), windows=self.windows
        )
        x_all = feat_df.reindex(ret_ser.index).fillna(0.0).to_numpy(dtype=np.float64)
        x_train = x_all[:-1]
        y_train = states[1:]

        unique_classes = np.unique(y_train)
        if len(unique_classes) >= 2 and len(x_train) >= 10:
            clf = self._build_classifier()
            try:
                clf.fit(x_train, y_train)
                pred_p = clf.predict_proba(x_all[-1:])[0]
                next_probs = np.zeros(k_eff, dtype=np.float64)
                classes = getattr(clf, "classes_", unique_classes)
                for c_idx, cls_val in enumerate(classes):
                    if 0 <= int(cls_val) < k_eff:
                        next_probs[int(cls_val)] = float(pred_p[c_idx])
                # Blend with current SJM smooth probability for temporal stability
                curr_p = reg_state.current_probs.to_numpy(dtype=np.float64)
                next_probs = 0.70 * next_probs + 0.30 * curr_p
                next_probs = next_probs / max(float(np.sum(next_probs)), 1e-12)
            except (ValueError, KeyError, RuntimeError):
                next_probs = reg_state.current_probs.to_numpy(dtype=np.float64)
        else:
            next_probs = reg_state.current_probs.to_numpy(dtype=np.float64)

        q_i = float(next_probs @ reg_means)
        # Law of total variance across regime mixture
        within_var = float(next_probs @ reg_vars)
        between_var = float(next_probs @ ((reg_means - q_i) ** 2))
        omega_i = max(within_var + between_var, 1e-5)

        if len(next_probs) < self.n_regimes:
            padded = np.zeros(self.n_regimes, dtype=np.float64)
            padded[: len(next_probs)] = next_probs
            next_probs = padded
        return q_i, omega_i, next_probs[: self.n_regimes]

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        assets = list(data.prices.columns)
        n_assets = len(assets)

        q_vec = np.zeros(n_assets, dtype=np.float64)
        omega_diag = np.zeros(n_assets, dtype=np.float64)
        probs_mat = np.zeros((n_assets, self.n_regimes), dtype=np.float64)

        for i, asset in enumerate(assets):
            q_i, om_i, p_i = self._forecast_single_asset(asset, data.prices[asset])
            q_vec[i] = q_i
            omega_diag[i] = om_i
            probs_mat[i] = p_i

        self.last_next_regime_probs_ = pd.DataFrame(
            probs_mat,
            index=assets,
            columns=[f"regime_{k}" for k in range(self.n_regimes)],
        )
        self.last_posterior_std_ = pd.Series(
            np.sqrt(omega_diag), index=assets, name="posterior_std"
        )

        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        Q = pd.Series(q_vec, index=assets, name="expected_returns")
        Omega = pd.DataFrame(np.diag(omega_diag), index=assets, columns=assets)
        return P, Q, Omega

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        _, q_ser, _ = self.generate(data, **kwargs)
        return q_ser
