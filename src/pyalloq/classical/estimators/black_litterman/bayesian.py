from typing import Any, Tuple

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import (
    BaseCovarianceEstimator,
    BaseReturnEstimator,
    BaseViewGenerator,
)

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance


class BlackLittermanEstimator:
    """
    Core Black-Litterman Bayesian return and covariance posterior estimator.
    Can be called directly via `.estimate(market_caps, cov_matrix, P, Q, Omega)`
    or via `.estimate_from_data(data, ...)` using `MarketData`.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        cov_estimator: BaseCovarianceEstimator | None = None,
        view_generator: BaseViewGenerator | None = None,
        P: pd.DataFrame | None = None,
        Q: pd.DataFrame | pd.Series | None = None,
        Omega: pd.DataFrame | None = None,
    ) -> None:
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.cov_estimator = cov_estimator or EmpiricalCovariance()
        self.view_generator = view_generator
        self.P = P
        self.Q = Q
        self.Omega = Omega

    def estimate(
        self,
        market_caps: pd.Series,
        cov_matrix: pd.DataFrame,
        P: pd.DataFrame,
        Q: pd.DataFrame | pd.Series,
        Omega: pd.DataFrame | None = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        assets = cov_matrix.columns
        market_caps = market_caps.reindex(assets).fillna(1.0)
        Sigma = cov_matrix.values
        P_val = P.reindex(columns=assets, fill_value=0.0).values
        Q_val = np.asarray(Q.values, dtype=float).reshape(-1, 1)

        mcap_sum = float(market_caps.sum())
        if mcap_sum <= 1e-12:
            market_weights = np.full((len(assets), 1), 1.0 / len(assets))
        else:
            market_weights = (
                (market_caps / mcap_sum).to_numpy(dtype=float).reshape(-1, 1)
            )
        Pi = self.risk_aversion * Sigma.dot(market_weights)

        if Omega is None:
            tau_Sigma = self.tau * Sigma
            omega_val = np.diag(np.diag(P_val.dot(tau_Sigma).dot(P_val.T)))
        else:
            omega_val = Omega.values

        tau_cov_inv = np.linalg.pinv(self.tau * Sigma)
        omega_inv = np.linalg.pinv(omega_val)

        term1 = np.linalg.pinv(tau_cov_inv + P_val.T.dot(omega_inv).dot(P_val))
        term2 = tau_cov_inv.dot(Pi) + P_val.T.dot(omega_inv).dot(Q_val)

        bl_mu = term1.dot(term2).flatten()
        bl_cov = Sigma + term1
        bl_cov = 0.5 * (bl_cov + bl_cov.T)

        return (
            pd.Series(bl_mu, index=assets, name="expected_returns"),
            pd.DataFrame(bl_cov, index=assets, columns=assets),
        )

    def estimate_from_data(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        P: pd.DataFrame | None = None,
        Q: pd.DataFrame | pd.Series | None = None,
        Omega: pd.DataFrame | None = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        assets = list(data.prices.columns)
        cov = (
            cov_matrix if cov_matrix is not None else self.cov_estimator.estimate(data)
        )

        if (
            data.cross_sectional is not None
            and "market_cap" in data.cross_sectional.columns
        ):
            mcaps = data.cross_sectional["market_cap"].reindex(assets).fillna(1.0)
        else:
            mcaps = pd.Series(1.0, index=assets)

        p_mat = P if P is not None else self.P
        q_vec = Q if Q is not None else self.Q
        omega_mat = Omega if Omega is not None else self.Omega

        if (p_mat is None or q_vec is None) and self.view_generator is not None:
            p_mat, q_vec, omega_mat = self.view_generator.generate(data)

        if p_mat is None or q_vec is None:
            # No active views: return CAPM prior equilibrium returns and prior covariance
            mcap_sum = float(mcaps.sum())
            w_mkt = (
                mcaps / mcap_sum
                if mcap_sum > 1e-12
                else pd.Series(1.0 / len(assets), index=assets)
            )
            pi = self.risk_aversion * cov.dot(w_mkt)
            return pd.Series(pi, index=assets, name="expected_returns"), cov

        return self.estimate(
            market_caps=mcaps,
            cov_matrix=cov,
            P=p_mat,
            Q=q_vec,
            Omega=omega_mat,
        )


class BlackLittermanReturnEstimator(BaseReturnEstimator):
    """
    Pipeline-compatible `BaseReturnEstimator` adapter for Black-Litterman posterior returns.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        cov_estimator: BaseCovarianceEstimator | None = None,
        view_generator: BaseViewGenerator | None = None,
        P: pd.DataFrame | None = None,
        Q: pd.DataFrame | pd.Series | None = None,
        Omega: pd.DataFrame | None = None,
    ) -> None:
        self._bl = BlackLittermanEstimator(
            tau=tau,
            risk_aversion=risk_aversion,
            cov_estimator=cov_estimator,
            view_generator=view_generator,
            P=P,
            Q=Q,
            Omega=Omega,
        )

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        post_mu, _ = self._bl.estimate_from_data(data, **kwargs)
        return post_mu


class BlackLittermanCovarianceEstimator(BaseCovarianceEstimator):
    """
    Pipeline-compatible `BaseCovarianceEstimator` adapter for Black-Litterman posterior covariance.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        cov_estimator: BaseCovarianceEstimator | None = None,
        view_generator: BaseViewGenerator | None = None,
        P: pd.DataFrame | None = None,
        Q: pd.DataFrame | pd.Series | None = None,
        Omega: pd.DataFrame | None = None,
    ) -> None:
        self._bl = BlackLittermanEstimator(
            tau=tau,
            risk_aversion=risk_aversion,
            cov_estimator=cov_estimator,
            view_generator=view_generator,
            P=P,
            Q=Q,
            Omega=Omega,
        )

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        _, post_cov = self._bl.estimate_from_data(data, **kwargs)
        return post_cov
