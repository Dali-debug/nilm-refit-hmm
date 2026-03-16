"""
Evaluation metrics for NILM disaggregation.

Metrics
-------
MAE   : Mean Absolute Error (W)
RMSE  : Root Mean Squared Error (W)
NRMSE : RMSE normalised by (P95 − P5) of ground truth
REE   : Relative Energy Error (dimensionless)
F1    : ON/OFF F1-score (based on ON threshold)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ON_THRESHOLD

logger = logging.getLogger(__name__)


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    return float(np.mean(np.abs(true[mask] - pred[mask]))) if mask.any() else float("nan")


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    mask = ~(np.isnan(true) | np.isnan(pred))
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))) if mask.any() else float("nan")


def nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    """RMSE normalised by (P95 − P5) of *true*.  Returns NaN if range is 0."""
    r = float(np.nanpercentile(true, 95) - np.nanpercentile(true, 5))
    if r == 0:
        return float("nan")
    return rmse(true, pred) / r


def ree(true: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    """Relative Energy Error: |E_pred − E_true| / (E_true + eps)."""
    mask = ~(np.isnan(true) | np.isnan(pred))
    e_true = float(np.sum(true[mask]))
    e_pred = float(np.sum(pred[mask]))
    return abs(e_pred - e_true) / (abs(e_true) + eps)


def f1_on_off(
    true: np.ndarray,
    pred: np.ndarray,
    threshold: float = 10.0,
) -> float:
    """Binary F1-score for ON/OFF state detection."""
    mask = ~(np.isnan(true) | np.isnan(pred))
    y_true = (true[mask] >= threshold).astype(int)
    y_pred = (pred[mask] >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate_appliance(
    true: pd.Series,
    pred: pd.Series,
    appliance: str,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for one appliance.

    Parameters
    ----------
    true : pd.Series
        Ground-truth power.
    pred : pd.Series
        Predicted power.
    appliance : str
    threshold : float or None
        ON threshold; defaults to config.ON_THRESHOLD[appliance].

    Returns
    -------
    dict with keys: MAE, RMSE, NRMSE, REE, F1
    """
    if threshold is None:
        threshold = ON_THRESHOLD.get(appliance, 10.0)

    # Align on common index
    common = true.index.intersection(pred.index)
    t = true.reindex(common).values.astype(float)
    p = pred.reindex(common).values.astype(float)

    return {
        "MAE":   mae(t, p),
        "RMSE":  rmse(t, p),
        "NRMSE": nrmse(t, p),
        "REE":   ree(t, p),
        "F1":    f1_on_off(t, p, threshold=threshold),
    }


def evaluate_all(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    appliances: Optional[List[str]] = None,
    house_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute metrics for every appliance and return a tidy DataFrame.

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground-truth DataFrame (columns = appliance names).
    pred_df : pd.DataFrame
        Predictions DataFrame.
    appliances : list[str] or None
        Subset to evaluate; defaults to intersection of columns.
    house_id : int or None
        If provided, adds a 'house' column.

    Returns
    -------
    pd.DataFrame  columns: house (opt), appliance, MAE, RMSE, NRMSE, REE, F1
    """
    if appliances is None:
        appliances = list(set(true_df.columns) & set(pred_df.columns))

    rows = []
    for app in appliances:
        if app not in true_df.columns or app not in pred_df.columns:
            continue
        metrics = evaluate_appliance(true_df[app], pred_df[app], app)
        row = {"appliance": app, **metrics}
        if house_id is not None:
            row["house"] = house_id
        rows.append(row)

    result = pd.DataFrame(rows)
    if house_id is not None and "house" in result.columns:
        cols = ["house", "appliance"] + [c for c in result.columns if c not in ("house", "appliance")]
        result = result[cols]
    return result
