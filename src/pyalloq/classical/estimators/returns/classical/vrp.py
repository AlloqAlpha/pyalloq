from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator


class VarianceRiskPremiumEstimator(BaseReturnEstimator):
    """
    Point-in-Time Variance & Crash-Skewness Risk Premium Return Estimator
    (`VarianceRiskPremiumEstimator`, Bollerslev, Tauchen & Zhou 2009; Bakshi, Kapadia & Madan 2003).

    Estimates expected excess asset returns from:
    1. **Variance Risk Premium (VRP)**:
           VRP_{i, t} = sigma_{i, t, IV}^2 - sigma_{i, t, RV}^2
       where `sigma_{i, t, IV}` is the Point-in-Time Model-Free Implied Volatility
       (`data.features["implied_vol"]`) and `sigma_{i, t, RV}` is the Lifted Heston
       rough realized volatility (`data.features["lifted_heston_vol"]`) or trailing
       realized volatility.
    2. **Crash-Skewness Risk Premium**:
           SkewPrem_{i, t} = -MFIS_{i, t} * sigma_{i, t, IV}
       where negative risk-neutral skewness (`MFIS_{i, t} < 0` from `data.features["implied_skew"]`)
       reflects expensive out-of-the-money put protection and commands a positive expected return
       compensation in equilibrium:
           E_t[r_{i, t+1}] = r_f + gamma_vrp * VRP_{i, t} + eta_skew * (-MFIS_{i, t}) * sigma_{i, t, IV}
                             + (1 - blend_weight) * mu_{hist, i}
    """

    def __init__(
        self,
        vrp_aversion: float = 1.5,
        skew_aversion: float = 0.08,
        blend_weight: float = 0.65,
        realized_window: int = 30,
        implied_vol_feature: str = "implied_vol",
        rough_vol_feature: str = "lifted_heston_vol",
        implied_skew_feature: str = "implied_skew",
        annualization_factor: float = 252.0,
    ) -> None:
        self.vrp_aversion = float(vrp_aversion)
        self.skew_aversion = float(skew_aversion)
        self.blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
        self.realized_window = int(max(5, realized_window))
        self.implied_vol_feature = implied_vol_feature
        self.rough_vol_feature = rough_vol_feature
        self.implied_skew_feature = implied_skew_feature
        self.annualization_factor = float(annualization_factor)
        self.last_vrp_: pd.Series | None = None

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        tickers = list(data.prices.columns)
        returns = data.prices.pct_change().dropna(how="all").fillna(0.0)
        win_ret = returns.iloc[-self.realized_window :]

        hist_mean = (
            returns.ewm(span=self.realized_window, min_periods=2)
            .mean()
            .iloc[-1]
            .reindex(tickers)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            * self.annualization_factor
        )

        if self.rough_vol_feature in data.features:
            rv_vec = (
                data.features[self.rough_vol_feature]
                .iloc[-1]
                .reindex(tickers)
                .fillna(0.15)
                .to_numpy(dtype=np.float64)
            )
        else:
            rv_vec = win_ret.std(ddof=1).reindex(tickers).fillna(0.01).to_numpy(
                dtype=np.float64
            ) * np.sqrt(self.annualization_factor)

        if self.implied_vol_feature in data.features:
            iv_vec = (
                data.features[self.implied_vol_feature]
                .iloc[-1]
                .reindex(tickers)
                .fillna(pd.Series(rv_vec, index=tickers))
                .to_numpy(dtype=np.float64)
            )
        else:
            iv_vec = rv_vec * 1.05

        if self.implied_skew_feature in data.features:
            skew_vec = (
                data.features[self.implied_skew_feature]
                .iloc[-1]
                .reindex(tickers)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            skew_vec = (
                win_ret.skew().reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
            )

        vrp_vec = (iv_vec**2) - (rv_vec**2)
        self.last_vrp_ = pd.Series(vrp_vec, index=tickers, name="variance_risk_premium")

        if isinstance(data.risk_free_rate, pd.Series):
            rf = float(data.risk_free_rate.iloc[-1])
        else:
            rf = float(data.risk_free_rate)

        option_implied_mu = (
            rf + self.vrp_aversion * vrp_vec + self.skew_aversion * (-skew_vec) * iv_vec
        )
        blended_mu = (
            self.blend_weight * option_implied_mu
            + (1.0 - self.blend_weight) * hist_mean
        )
        return pd.Series(blended_mu, index=tickers, name="expected_returns")
