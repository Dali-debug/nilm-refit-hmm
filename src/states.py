"""
states.py
─────────
State labeling helpers for supervised HMM training.

Converts a continuous power time series into discrete state labels:
  - State 0  → appliance OFF (power ≤ on_threshold)
  - States 1…k → ON sub-levels, obtained by KMeans clustering on ON values

Usage
-----
    states, centroids = label_states(series, on_threshold=1000, n_states=2)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def label_states(
    series: pd.Series,
    on_threshold: float,
    n_states: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each time step with a discrete HMM state index.

    Parameters
    ----------
    series:
        Power time series (W), index = DatetimeIndex.
    on_threshold:
        Minimum power (W) to consider the appliance ON.
    n_states:
        Total number of states including the OFF state.
        If ``n_states == 2`` the ON state is a single level.
        If ``n_states > 2`` ON values are further clustered into
        ``n_states - 1`` levels.
    random_state:
        Seed for KMeans reproducibility.

    Returns
    -------
    labels : np.ndarray of shape (T,)
        Integer state label per time step (0 = OFF).
    centroids : np.ndarray of shape (n_states,)
        Mean power for each state (sorted ascending).
    """
    values = series.fillna(0).values.astype(float)
    labels = np.zeros(len(values), dtype=int)

    on_mask = values > on_threshold
    n_on_states = n_states - 1  # states for ON levels

    if n_on_states < 1:
        raise ValueError("n_states must be >= 2.")

    on_values = values[on_mask]

    if len(on_values) == 0:
        # No ON events found — return all OFF
        logger.warning("No ON events found (threshold=%.1f W). All states = OFF.", on_threshold)
        centroids = np.zeros(n_states)
        return labels, centroids

    if n_on_states == 1:
        labels[on_mask] = 1
        on_centroid = np.mean(on_values)
        centroids = np.array([0.0, on_centroid])
    else:
        # KMeans on ON values to create sub-levels
        km = KMeans(n_clusters=n_on_states, random_state=random_state, n_init="auto")
        km.fit(on_values.reshape(-1, 1))
        raw_labels = km.labels_  # 0…(n_on_states-1), arbitrary order

        # Re-order clusters by ascending centroid power
        order = np.argsort(km.cluster_centers_.ravel())
        remap = np.empty_like(order)
        remap[order] = np.arange(n_on_states)
        remapped = remap[raw_labels] + 1  # 1-indexed (0 = OFF)

        labels[on_mask] = remapped
        sorted_centers = np.sort(km.cluster_centers_.ravel())
        centroids = np.concatenate([[0.0], sorted_centers])

    return labels, centroids
