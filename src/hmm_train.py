"""
Train one GaussianHMM per appliance.

Usage
-----
    from src.hmm_train import train_appliance_model, save_model, load_model
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.config import N_STATES, RANDOM_SEED
from src.states import build_state_definitions, state_defs_to_hmm_params
from src.utils import stack_segments

logger = logging.getLogger(__name__)


def train_appliance_model(
    series_list: List[pd.Series],
    appliance: str,
    n_states: Optional[int] = None,
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = RANDOM_SEED,
) -> GaussianHMM:
    """
    Train a GaussianHMM for one appliance from multiple training houses.

    Parameters
    ----------
    series_list : list[pd.Series]
        One series per training house (clean, resampled).
    appliance : str
        Appliance name.
    n_states : int or None
        Number of HMM states.
    n_iter : int
        Max EM iterations.
    tol : float
        Convergence tolerance.
    random_state : int
        Random seed.

    Returns
    -------
    hmmlearn.hmm.GaussianHMM  (fitted)
    """
    if n_states is None:
        n_states = N_STATES.get(appliance, 2)

    # Stack all non-NaN segments from training houses
    obs, lengths = stack_segments(*series_list, min_length=n_states + 1)

    if len(obs) == 0 or sum(lengths) == 0:
        raise ValueError(
            f"No valid training data for appliance '{appliance}'."
        )

    logger.info(
        "Training HMM for '%s': %d states, %d observations from %d segments.",
        appliance, n_states, len(obs), len(lengths),
    )

    # Build state definitions for initialisation (using all combined data)
    combined = pd.Series(obs.flatten())
    state_defs = build_state_definitions(combined, appliance, n_states=n_states)
    init_means, init_covars = state_defs_to_hmm_params(state_defs)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        init_params="st",   # initialise start prob and transitions
        params="stmc",      # learn all
    )

    # Override means and covars with data-driven init
    model.means_  = init_means
    model.covars_ = init_covars

    model.fit(obs, lengths)

    logger.info("Fitted HMM for '%s' | score: %.4f", appliance, model.score(obs, lengths))
    return model


def save_model(model: GaussianHMM, path: str) -> None:
    """Persist a trained model with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> GaussianHMM:
    """Load a model saved with save_model()."""
    return joblib.load(path)


def train_all_appliances(
    house_data: Dict[int, pd.DataFrame],
    train_houses: List[int],
    appliances: Optional[List[str]] = None,
    models_dir: str = "models/hmm/loho",
    test_house: Optional[int] = None,
    **kwargs,
) -> Dict[str, GaussianHMM]:
    """
    Convenience wrapper: train one model per appliance using *train_houses*.

    Parameters
    ----------
    house_data : dict[int, pd.DataFrame]
        Pre-loaded and cleaned DataFrames keyed by house ID.
    train_houses : list[int]
        Houses to use for training.
    appliances : list[str] or None
        Target appliances.  Defaults to columns present in all houses.
    models_dir : str
        Where to save models.
    test_house : int or None
        For naming the save directory (LOHO scenario).

    Returns
    -------
    dict[str, GaussianHMM]
    """
    from src.config import TARGET_APPLIANCES
    if appliances is None:
        appliances = TARGET_APPLIANCES

    models: Dict[str, GaussianHMM] = {}
    subdir = f"test_house_{test_house:02d}" if test_house is not None else "all"

    for app in appliances:
        series_list = []
        for h in train_houses:
            df = house_data.get(h)
            if df is None or app not in df.columns:
                continue
            series_list.append(df[app])

        if not series_list:
            logger.warning("Skipping '%s': no training data.", app)
            continue

        try:
            mdl = train_appliance_model(series_list, app, **kwargs)
            save_path = str(Path(models_dir) / subdir / f"{app}.pkl")
            save_model(mdl, save_path)
            models[app] = mdl
        except ValueError as exc:
            logger.error("Could not train '%s': %s", app, exc)

    return models
