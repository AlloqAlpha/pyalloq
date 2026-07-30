import numpy as np
import pandas as pd
from typing import Tuple

class BlackLittermanEstimator:
    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:
        self.tau = tau
        self.risk_aversion = risk_aversion

    def estimate(
        self,
        market_caps: pd.Series,
        cov_matrix: pd.DataFrame,
        P: pd.DataFrame,
        Q: pd.DataFrame,
        Omega: pd.DataFrame | None = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:

        assets = cov_matrix.columns
        market_caps = market_caps.loc[assets]
        Sigma = cov_matrix.values
        P_val = P.values
        Q_val = Q.values

        market_weights = (market_caps / market_caps.sum()).values.reshape(-1, 1)
        Pi = self.risk_aversion * Sigma.dot(market_weights)

        if Omega is None:
            tau_Sigma = self.tau * Sigma
            omega_val = np.diag(np.diag(P_val.dot(tau_Sigma).dot(P_val.T)))
        else:
            omega_val = omega_val.values

        tau_cov_inv = np.linalg.inv(self.tau * Sigma)
        omega_inv = np.linalg.inv(omega_val)

        term1 = np.linalg.inv(tau_cov_inv + P_val.T.dot(omega_inv).dot(P_val))
        term2 = tau_cov_inv.dot(Pi) + P_val.T.dot(omega_inv).dot(Q_val)

        bl_mu = term1.dot(term2).flatten()
        bl_cov = Sigma + term1

        return (
            pd.Series(bl_mu, index=assets),
            pd.DataFrame(bl_cov, index=assets, columns=assets)
        )