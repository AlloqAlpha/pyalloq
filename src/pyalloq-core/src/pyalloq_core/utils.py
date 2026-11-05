import numpy as np
import pandas as pd

from .enums import DataFrequency


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Helper to convert covariance matrix to correlation matrix."""
    vols = np.sqrt(np.diag(cov))
    outer_vols = np.outer(vols, vols)
    corr = cov / outer_vols
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def resolve_annualization_factor(
    returns: pd.DataFrame | pd.Series | None = None,
    frequency: DataFrequency | str | int | float | None = None,
) -> float:
    """Determines annualization factor from DataFrequency Enum, int/float, string, or pandas index."""
    if isinstance(frequency, DataFrequency):
        return float(frequency.value)
    if isinstance(frequency, (int, float)):
        return float(frequency)
    if isinstance(frequency, str):
        try:
            return float(DataFrequency[frequency.upper()].value)
        except KeyError:
            pass

    if returns is not None and isinstance(returns.index, pd.DatetimeIndex):
        inferred = pd.infer_freq(returns.index)
        if inferred:
            if "W" in inferred:
                return float(DataFrequency.WEEKLY.value)
            if "M" in inferred or "MS" in inferred:
                return float(DataFrequency.MONTHLY.value)
            if "B" in inferred or "D" in inferred:
                return float(DataFrequency.DAILY.value)

    return float(DataFrequency.DAILY.value)
