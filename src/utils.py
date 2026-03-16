"""
Common utility helpers for the NILM pipeline.
"""

import random
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_continuous_segments(
    series: pd.Series,
    min_length: int = 2,
) -> Iterator[pd.Series]:
    """
    Yield contiguous non-NaN segments of *series*.

    Parameters
    ----------
    series : pd.Series
        1-D time series, possibly containing NaN.
    min_length : int
        Minimum segment length (in samples) to yield.

    Yields
    ------
    pd.Series
        Sub-series with no NaN values.
    """
    notna = series.notna()
    # Label consecutive groups
    group_ids = (notna != notna.shift()).cumsum()
    for _, grp in series.groupby(group_ids):
        if grp.notna().all() and len(grp) >= min_length:
            yield grp


def stack_segments(
    *series_list: pd.Series,
    min_length: int = 2,
) -> Tuple[np.ndarray, list]:
    """
    Stack continuous non-NaN segments from multiple series into arrays
    suitable for hmmlearn (observations + lengths).

    Parameters
    ----------
    *series_list : pd.Series
        One or more series to concatenate after segmenting.
    min_length : int
        Minimum segment length.

    Returns
    -------
    obs : np.ndarray, shape (N, 1)
    lengths : list[int]
    """
    all_obs   = []
    lengths   = []
    for s in series_list:
        for seg in get_continuous_segments(s, min_length=min_length):
            vals = seg.values.reshape(-1, 1)
            all_obs.append(vals)
            lengths.append(len(seg))
    if not all_obs:
        return np.empty((0, 1)), []
    return np.vstack(all_obs), lengths
