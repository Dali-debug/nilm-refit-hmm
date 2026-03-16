"""
Greedy additive disaggregation.

Algorithm
---------
For each appliance (in INFERENCE_ORDER):
  1. Decode state sequence from current residual via Viterbi.
  2. Reconstruct power from state means.
  3. Clip to residual (cannot exceed remaining power).
  4. Subtract from residual.

This is an approximation: errors accumulate for later appliances, but it is
simple, interpretable, and works well for appliances with distinct signatures.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.config import INFERENCE_ORDER

logger = logging.getLogger(__name__)


def predict_appliance(
    residual: pd.Series,
    model: GaussianHMM,
) -> pd.Series:
    """
    Viterbi decode on *residual* and reconstruct power from state means.

    Parameters
    ----------
    residual : pd.Series
        Current residual mains power (Watts).
    model : GaussianHMM
        Fitted model for the appliance.

    Returns
    -------
    pd.Series
        Reconstructed power, same index as *residual*.
    """
    obs = residual.values.reshape(-1, 1).astype(float)
    # Replace NaN with 0 for Viterbi (won't affect output masking)
    nan_mask = np.isnan(obs.flatten())
    obs[nan_mask] = 0.0

    lengths = [len(obs)]
    _, state_seq = model.decode(obs, lengths=lengths, algorithm="viterbi")

    state_means = model.means_.flatten()
    power_hat = state_means[state_seq]

    result = pd.Series(power_hat, index=residual.index, dtype=float)
    # Restore NaN where input was NaN
    result[nan_mask] = np.nan
    return result


def greedy_disaggregate(
    mains: pd.Series,
    models: Dict[str, GaussianHMM],
    order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Iteratively disaggregate *mains* into per-appliance power estimates.

    Parameters
    ----------
    mains : pd.Series
        Aggregate mains power (Watts), datetime-indexed.
    models : dict[str, GaussianHMM]
        Fitted models keyed by appliance name.
    order : list[str] or None
        Disaggregation order.  Defaults to config.INFERENCE_ORDER.

    Returns
    -------
    pd.DataFrame
        One column per appliance, same index as *mains*.
    """
    if order is None:
        order = INFERENCE_ORDER

    residual = mains.copy().clip(lower=0)
    predictions: Dict[str, pd.Series] = {}

    for app in order:
        if app not in models:
            logger.debug("No model for '%s'; skipping.", app)
            continue

        logger.info("Disaggregating '%s'...", app)
        power_hat = predict_appliance(residual, models[app])

        # Clip to non-negative residual
        power_hat = power_hat.clip(lower=0)
        power_hat = power_hat.where(power_hat <= residual.fillna(0), other=residual.fillna(0))

        predictions[app] = power_hat
        residual = (residual - power_hat).clip(lower=0)
        logger.info("  residual mean after '%s': %.1f W", app, residual.mean())

    return pd.DataFrame(predictions, index=mains.index)
