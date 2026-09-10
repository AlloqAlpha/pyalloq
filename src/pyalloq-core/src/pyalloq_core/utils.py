import numpy as np


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Helper to convert covariance matrix to correlation matrix."""
    vols = np.sqrt(np.diag(cov))
    outer_vols = np.outer(vols, vols)
    corr = cov / outer_vols
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr
