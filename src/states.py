"""
Build discrete state definitions from submeter training data.

Strategy
--------
* State 0 is always OFF (power ≈ 0).
* For n_states > 2, ON samples are clustered with KMeans to produce
  power level sub-states, ordered by mean power.
* Returns initial means / covariances for GaussianHMM.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import ON_THRESHOLD, N_STATES, RANDOM_SEED


def build_state_definitions(
    series: pd.Series,
    appliance: str,
    n_states: int | None = None,
    on_threshold: float | None = None,
    random_state: int = RANDOM_SEED,
) -> Dict:
    """
    Compute state means and variances for a single appliance.

    Parameters
    ----------
    series : pd.Series
        Clean submeter power series (Watts, no NaN).
    appliance : str
        Appliance name (used to look up defaults).
    n_states : int or None
        Number of HMM states.  Defaults to config.N_STATES[appliance].
    on_threshold : float or None
        Minimum power (W) to classify a sample as ON.

    Returns
    -------
    dict with keys:
        n_states  : int
        means     : np.ndarray  shape (n_states,)
        vars      : np.ndarray  shape (n_states,)
        labels    : list[str]   e.g. ["OFF", "ON"] or ["OFF", "LOW", "HIGH"]
    """
    if n_states is None:
        n_states = N_STATES.get(appliance, 2)
    if on_threshold is None:
        on_threshold = ON_THRESHOLD.get(appliance, 10.0)

    values = series.dropna().values.astype(float)

    # OFF state
    off_vals = values[values < on_threshold]
    off_mean = float(np.mean(off_vals)) if len(off_vals) else 0.0
    off_var  = float(np.var(off_vals))  if len(off_vals) else 1.0
    off_var  = max(off_var, 1.0)  # avoid zero variance

    if n_states == 2:
        on_vals = values[values >= on_threshold]
        on_mean = float(np.mean(on_vals)) if len(on_vals) else on_threshold * 10
        on_var  = float(np.var(on_vals))  if len(on_vals) else on_mean * 0.1
        on_var  = max(on_var, 1.0)
        return {
            "n_states": 2,
            "means":    np.array([off_mean, on_mean]),
            "vars":     np.array([off_var,  on_var]),
            "labels":   ["OFF", "ON"],
        }

    # n_states > 2 → cluster ON samples
    on_vals = values[values >= on_threshold]
    n_clusters = n_states - 1  # excluding OFF

    if len(on_vals) < n_clusters:
        # Not enough ON samples → fallback to 2 states
        on_mean = float(np.mean(on_vals)) if len(on_vals) else on_threshold * 10
        on_var  = float(np.var(on_vals))  if len(on_vals) else on_mean * 0.1
        on_var  = max(on_var, 1.0)
        means = np.array([off_mean] + [off_mean + (on_mean - off_mean) * i / (n_states - 1)
                                        for i in range(1, n_states)])
        vars_ = np.array([off_var] + [max(on_var, 1.0)] * (n_states - 1))
        labels = ["OFF"] + [f"ON_{i}" for i in range(1, n_states)]
        return {"n_states": n_states, "means": means, "vars": vars_, "labels": labels}

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(on_vals.reshape(-1, 1))

    # Sort clusters by mean power
    cluster_means = km.cluster_centers_.flatten()
    order = np.argsort(cluster_means)

    means = np.empty(n_states)
    vars_ = np.empty(n_states)
    means[0] = off_mean
    vars_[0] = off_var
    labels = ["OFF"]

    for rank, orig_idx in enumerate(order):
        mask   = km.labels_ == orig_idx
        c_vals = on_vals[mask]
        means[rank + 1] = cluster_means[orig_idx]
        vars_[rank + 1] = max(float(np.var(c_vals)), 1.0)
        labels.append(f"ON_{rank + 1}" if n_clusters > 1 else "ON")

    return {"n_states": n_states, "means": means, "vars": vars_, "labels": labels}


def state_defs_to_hmm_params(
    state_defs: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert state definitions to hmmlearn-compatible means_ / covars_.

    Returns
    -------
    means_  : np.ndarray  shape (n_states, 1)
    covars_ : np.ndarray  shape (n_states, 1, 1)
    """
    means  = state_defs["means"].reshape(-1, 1)
    covars = state_defs["vars"].reshape(-1, 1, 1)
    return means, covars
