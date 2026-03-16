"""
Data cleaning and resampling for REFIT data.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import RESAMPLE_RULE, INTERP_LIMIT, POWER_CAP

logger = logging.getLogger(__name__)


def clean_resample(
    df: pd.DataFrame,
    resample_rule: str = RESAMPLE_RULE,
    interp_limit: int = INTERP_LIMIT,
    power_cap: Optional[Dict[str, float]] = None,
    report: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Clean and resample a standardised REFIT DataFrame.

    Steps:
    1. Replace negative values with NaN.
    2. Cap outliers per column (configurable power cap).
    3. Resample to *resample_rule* using mean aggregation.
    4. Interpolate short gaps (≤ interp_limit consecutive NaNs).

    Parameters
    ----------
    df : pd.DataFrame
        Standardised DataFrame (columns: mains, kettle, …).
    resample_rule : str
        Pandas offset alias, default "1min".
    interp_limit : int
        Maximum consecutive NaN periods to fill by interpolation.
    power_cap : dict or None
        Column-level power caps (Watts).  Falls back to config.POWER_CAP.
    report : bool
        If True, also return a stats dict.

    Returns
    -------
    cleaned : pd.DataFrame
    stats : dict or None  (only when report=True)
    """
    if power_cap is None:
        power_cap = POWER_CAP

    stats: Dict = {}

    # ── 1. Negative → NaN ────────────────────────────────────────────────────
    neg_counts = (df < 0).sum()
    df = df.clip(lower=0)
    df = df.where(df >= 0, other=np.nan)
    stats["neg_replaced"] = neg_counts.to_dict()

    # ── 2. Cap outliers ───────────────────────────────────────────────────────
    cap_counts: Dict[str, int] = {}
    for col in df.columns:
        cap = power_cap.get(col, power_cap.get("mains", 20_000))
        mask = df[col] > cap
        cap_counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    stats["outliers_capped"] = cap_counts

    # Additional quantile-based cap (99th percentile per column)
    for col in df.columns:
        q99 = df[col].quantile(0.99)
        if q99 > 0:
            mask = df[col] > q99 * 2
            df.loc[mask, col] = np.nan

    # ── 3. Resample ───────────────────────────────────────────────────────────
    df_resampled = df.resample(resample_rule).mean()
    stats["rows_before"] = len(df)
    stats["rows_after"]  = len(df_resampled)

    # ── 4. Interpolate short gaps ─────────────────────────────────────────────
    missing_before = df_resampled.isna().sum()
    df_clean = df_resampled.interpolate(
        method="time",
        limit=interp_limit,
        limit_direction="forward",
    )
    missing_after = df_clean.isna().sum()
    stats["missing_before_interp"] = missing_before.to_dict()
    stats["missing_after_interp"]  = missing_after.to_dict()
    stats["interpolated"]          = (missing_before - missing_after).to_dict()

    logger.info(
        "clean_resample: %d → %d rows; interpolated %s points.",
        stats["rows_before"],
        stats["rows_after"],
        sum(stats["interpolated"].values()),
    )

    if report:
        return df_clean, stats
    return df_clean, None
