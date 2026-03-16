"""
inference.py
────────────
Greedy iterative-additive disaggregation.

Algorithm
---------
Given ``mains`` and a list of (appliance_name, model, centroids) tuples
sorted by ``order`` (easiest first):

    residual(t) = mains(t)
    for each appliance k:
        state_seq = Viterbi( residual(t) | model_k )
        power_k(t) = centroids_k[ state_seq(t) ]
        residual(t) = max(0, residual(t) − power_k(t))

Returns a DataFrame with one column per appliance.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.hmm_model import predict_states

logger = logging.getLogger(__name__)


def disaggregate(
    mains: pd.Series,
    appliance_models: list[tuple[str, GaussianHMM, np.ndarray]],
) -> pd.DataFrame:
    """Greedy iterative-additive NILM disaggregation.

    Parameters
    ----------
    mains:
        Aggregate power series (W), 1-minute resolution, DatetimeIndex.
    appliance_models:
        List of ``(name, model, centroids)`` tuples, already sorted by
        greedy subtraction order (simplest appliance first).

    Returns
    -------
    pd.DataFrame
        Columns = appliance names, index = same as ``mains``.
        Values = estimated power (W) for each time step.
    """
    residual = mains.fillna(0).values.astype(float).copy()
    results: dict[str, np.ndarray] = {}

    for name, model, centroids in appliance_models:
        logger.info("Disaggregating '%s' …", name)

        # Build a temporary Series from the current residual
        obs_series = pd.Series(residual, index=mains.index)

        # Viterbi on residual
        states = predict_states(model, obs_series)

        # Reconstruct estimated power from state centroids
        power_est = centroids[states].astype(float)

        # Clip to residual (cannot exceed what is left)
        power_est = np.minimum(power_est, residual)
        power_est = np.maximum(power_est, 0.0)

        results[name] = power_est

        # Update residual
        residual = np.maximum(0.0, residual - power_est)

    return pd.DataFrame(results, index=mains.index)
