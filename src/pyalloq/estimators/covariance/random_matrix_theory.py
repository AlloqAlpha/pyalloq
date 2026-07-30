import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from .base import BaseCovarianceEstimator

class RandomMatrixTheoryEstimator(BaseCovarianceEstimator):
    def __init__(self, denoise: bool = True, detone: bool = False):
        self.denoise = denoise
        self.detone = detone

    def estimate(self, prices: pd.DataFrame)  -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        T, N = returns.shape

        cov_empirical = returns.cov() * 252
        std_devs = np.sqrt(np.diag(cov_empirical))
        corr_matrix = returns.corr().values

        evals, evecs = np.linalg.eigh(corr_matrix)

        indices = evals.argsort()[::-1]
        evals = evals[indices]
        evecs = evecs[:, indices]

        if self.denoise:
            q = T / N
            eMax = (1 + np.sqrt(1 / q))**2

            n_factors = evals[evals > eMax].shape[0]

            if n_factors < N:
                noise_average = np.sum(evals[n_factors]) / (N - n_factors)
                evals[n_factors:] = noise_average

        if self.detone:
            evals = evals[1:]
            evecs = evecs[:, 1:]

        corr_reconstructed = np.dot(evecs, np.dot(np.diag(evals, evecs.T)))
        np.fill_diagonal(corr_reconstructed, 1.0)

        D = np.diag(std_devs)
        cov_reconstructed = np.dot(D, np.dot(corr_reconstructed, D))

        corr_reconstructed = (cov_reconstructed + cov_reconstructed.T) / 2

        return pd.DataFrame(
            cov_reconstructed, 
            index=prices.columns, 
            columns=prices.columns
        )