"""
hmm_model.py
────────────
Train, save, and load a GaussianHMM per appliance using hmmlearn.

Training is supervised: observations come from the appliance sub-meter
time series, state labels from ``states.label_states()``.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.states import label_states

logger = logging.getLogger(__name__)


def train_hmm(
    series: pd.Series,
    on_threshold: float,
    n_states: int,
    covariance_type: str = "full",
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[GaussianHMM, np.ndarray]:
    """Train a GaussianHMM on a single appliance time series.

    The model is initialised with emission means derived from ``label_states``
    (supervised initialisation) and then refined with the Baum-Welch algorithm.

    Parameters
    ----------
    series:
        Appliance power time series (W), DatetimeIndex, 1-minute resolution.
    on_threshold, n_states:
        Passed to :func:`src.states.label_states`.
    covariance_type, n_iter, tol, random_state:
        GaussianHMM hyper-parameters.

    Returns
    -------
    model : GaussianHMM
        Fitted model.
    centroids : np.ndarray
        State mean power values (shape ``(n_states,)``).
    """
    values = series.fillna(0).values.astype(float).reshape(-1, 1)
    state_labels, centroids = label_states(series, on_threshold, n_states, random_state)

    # ── Supervised initialisation of means ──────────────────────────────────
    init_means = centroids.reshape(-1, 1)
    init_covars = np.array(
        [
            max(
                float(np.var(values[state_labels == s])) if np.any(state_labels == s) else 1.0,
                1.0,
            )
            for s in range(n_states)
        ]
    ).reshape(n_states, 1, 1)

    # ── Transition matrix: favour staying in same state ──────────────────────
    stay_prob = 0.9
    off_diag = (1.0 - stay_prob) / max(n_states - 1, 1)
    init_transmat = np.full((n_states, n_states), off_diag)
    np.fill_diagonal(init_transmat, stay_prob)
    init_transmat = init_transmat / init_transmat.sum(axis=1, keepdims=True)

    # ── Start probabilities: mostly OFF ─────────────────────────────────────
    init_startprob = np.full(n_states, 0.1 / max(n_states - 1, 1))
    init_startprob[0] = 0.9
    init_startprob = init_startprob / init_startprob.sum()

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        init_params="",   # use our supervised initialisation, not hmmlearn defaults
    )
    model.startprob_ = init_startprob
    model.transmat_ = init_transmat
    model.means_ = init_means
    model.covars_ = init_covars

    model.fit(values)
    logger.info(
        "HMM trained — n_states=%d, converged=%s, log-likelihood=%.2f",
        n_states,
        model.monitor_.converged,
        model.score(values),
    )
    return model, centroids


def predict_states(model: GaussianHMM, observations: pd.Series) -> np.ndarray:
    """Run Viterbi decoding on an observation series.

    Parameters
    ----------
    model:
        Trained GaussianHMM.
    observations:
        Power series to decode (W).  NaNs are replaced by 0.

    Returns
    -------
    np.ndarray of int
        Predicted state sequence.
    """
    obs = observations.fillna(0).values.astype(float).reshape(-1, 1)
    return model.predict(obs)


def save_model(model: GaussianHMM, path: str | Path) -> None:
    """Pickle a trained HMM model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> GaussianHMM:
    """Load a pickled HMM model from disk."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
