import numpy as np


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Helper to convert covariance matrix to correlation matrix."""
    vols = np.sqrt(np.diag(cov))
    outer_vols = np.outer(vols, vols)
    return cov / outer_vols
