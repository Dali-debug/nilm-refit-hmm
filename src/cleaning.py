"""
cleaning.py
───────────
Data cleaning, outlier removal, and interpolation for NILM time series.

Steps applied per column:
  1. Remove duplicate timestamps (keep first).
  2. Replace negative power readings with NaN.
  3. Remove extreme outliers (above ``outlier_quantile * 1.5``).
  4. Resample to target frequency (mean within each bin).
  5. Time-based interpolation limited to ``interp_limit`` consecutive steps.
  6. Flag columns with too many remaining NaN values.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def remove_negatives(series: pd.Series) -> pd.Series:
    """Replace negative values with NaN."""
    return series.where(series >= 0, other=np.nan)


def remove_outliers(series: pd.Series, quantile: float = 0.999) -> pd.Series:
    """Replace values above ``quantile * 1.5`` with NaN (per-series cap)."""
    cap = series.quantile(quantile) * 1.5
    if np.isnan(cap) or cap <= 0:
        return series
    return series.where(series <= cap, other=np.nan)


def interpolate_limited(
    series: pd.Series,
    method: str = "time",
    limit: int = 5,
) -> pd.Series:
    """Interpolate small gaps (≤ ``limit`` consecutive NaNs)."""
    return series.interpolate(method=method, limit=limit, limit_direction="both")


def clean_dataframe(
    df: pd.DataFrame,
    resample_freq: str = "1min",
    outlier_quantile: float = 0.999,
    interp_method: str = "time",
    interp_limit: int = 5,
    max_missing_ratio: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline for a single house DataFrame.

    Parameters
    ----------
    df:
        Raw house DataFrame with DatetimeIndex.
    resample_freq:
        Pandas offset alias for resampling (default ``"1min"`` = 1 minute).
    outlier_quantile:
        Quantile used for the outlier cap (default 0.999).
    interp_method:
        Pandas interpolation method (default ``"time"``).
    interp_limit:
        Maximum number of consecutive missing periods to fill.
    max_missing_ratio:
        If a column still has more than this fraction of NaNs after
        interpolation it is returned as-is but flagged in the report.

    Returns
    -------
    cleaned_df:
        Resampled, cleaned DataFrame.
    report:
        Dictionary with per-column statistics (missing ratios, etc.).
    """
    # 1. Drop duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    report: dict = {}

    # 2–3. Per-column: negatives + outliers
    for col in df.columns:
        original_missing = df[col].isna().mean()
        df[col] = remove_negatives(df[col])
        df[col] = remove_outliers(df[col], quantile=outlier_quantile)
        report[col] = {"missing_before": original_missing}

    # 4. Resample to target frequency
    df = df.resample(resample_freq).mean()

    # 5. Interpolate small gaps
    for col in df.columns:
        missing_before_interp = df[col].isna().mean()
        df[col] = interpolate_limited(df[col], method=interp_method, limit=interp_limit)
        missing_after_interp = df[col].isna().mean()

        report[col].update(
            {
                "missing_before_interp": missing_before_interp,
                "missing_after_interp": missing_after_interp,
                "too_many_missing": missing_after_interp > max_missing_ratio,
            }
        )
        if missing_after_interp > max_missing_ratio:
            logger.warning(
                "Column '%s' has %.1f%% missing after interpolation.",
                col,
                missing_after_interp * 100,
            )

    return df, report
