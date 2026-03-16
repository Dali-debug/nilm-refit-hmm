"""
metrics.py
──────────
Evaluation KPIs for NILM disaggregation (per appliance, per house).

Metrics implemented
───────────────────
- MAE   : Mean Absolute Error (W)
- RMSE  : Root Mean Squared Error (W)
- NRMSE : Normalised RMSE (fraction), normalised by the (P95 − P5) range of
          the ground-truth signal (robust to outliers)
- REE   : Relative Energy Error = |E_pred − E_true| / (E_true + ε)
- F1    : Binary ON/OFF F1-score (positive class = ON)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


_MIN_RANGE = 1e-6  # minimum P95-P5 range to compute NRMSE


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by the P95-P5 range of the ground-truth."""
    rng = np.percentile(y_true, 95) - np.percentile(y_true, 5)
    if rng < _MIN_RANGE:
        return float("nan")
    return rmse(y_true, y_pred) / rng


def ree(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-3,
) -> float:
    """Relative Energy Error over the evaluation window."""
    e_true = float(np.sum(y_true))
    e_pred = float(np.sum(y_pred))
    return abs(e_pred - e_true) / (e_true + epsilon)


def f1_on_off(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    on_threshold: float,
) -> float:
    """Binary F1 score for ON/OFF state detection."""
    t_bin = (y_true > on_threshold).astype(int)
    p_bin = (y_pred > on_threshold).astype(int)
    return float(f1_score(t_bin, p_bin, zero_division=0))


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    on_threshold: float,
    energy_epsilon: float = 1e-3,
) -> dict[str, float]:
    """Compute all KPIs for a single (appliance, house) pair.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted power series aligned on the same index.
    on_threshold:
        Threshold (W) for binarising into ON/OFF.
    energy_epsilon:
        Small constant to avoid division by zero in REE.

    Returns
    -------
    dict with keys: mae, rmse, nrmse, ree, f1
    """
    # Align and fill NaN with 0
    y_true, y_pred = y_true.align(y_pred, join="inner")
    t = y_true.fillna(0).values.astype(float)
    p = y_pred.fillna(0).values.astype(float)

    return {
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "nrmse": nrmse(t, p),
        "ree": ree(t, p, epsilon=energy_epsilon),
        "f1": f1_on_off(t, p, on_threshold),
    }


def metrics_table(
    results: list[dict],
) -> pd.DataFrame:
    """Convert a list of per-appliance metric dicts into a summary DataFrame.

    Each dict should have keys: ``house``, ``appliance``, ``mae``, ``rmse``,
    ``nrmse``, ``ree``, ``f1``.
    """
    df = pd.DataFrame(results)
    return df.set_index(["house", "appliance"]).sort_index()
