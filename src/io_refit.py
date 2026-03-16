"""
Robust loader for REFIT House_*.csv files.

Expected CSV columns:
    Time, Unix, Aggregate, Appliance1 ... Appliance9
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_house(
    house_id: int,
    raw_dir: str = "data/raw",
    tz: str = "Europe/London",
) -> pd.DataFrame:
    """
    Load a single REFIT house CSV and return a timezone-aware DataFrame
    indexed by datetime with float64 columns (Watts).

    Parameters
    ----------
    house_id : int
        House number (1–21).
    raw_dir : str
        Directory containing House_*.csv files.
    tz : str
        Timezone for the timestamps.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeTzAware
        Columns: Unix, Aggregate, Appliance1 … Appliance9
    """
    raw_path = Path(raw_dir)
    csv_path = raw_path / f"House_{house_id}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_path.resolve()}\n"
            "Place REFIT CSV files under data/raw/ (not committed to git)."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"House CSV not found: {csv_path.resolve()}\n"
            "Make sure House_{house_id}.csv is present in data/raw/."
        )

    logger.info("Loading House %d from %s", house_id, csv_path)

    df = pd.read_csv(
        csv_path,
        header=0,
        parse_dates=["Time"],
        low_memory=False,
    )

    # Rename first two columns robustly
    df.columns = [c.strip() for c in df.columns]

    # Parse Unix column as fallback datetime if Time failed
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        logger.warning("Time column parse failed; falling back to Unix column.")
        df["Time"] = pd.to_datetime(df["Unix"], unit="s", utc=True).dt.tz_convert(tz)
    else:
        # Localise / convert
        if df["Time"].dt.tz is None:
            df["Time"] = df["Time"].dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            df["Time"] = df["Time"].dt.tz_convert(tz)

    df = df.set_index("Time").sort_index()

    # Remove duplicate timestamps (keep first)
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        logger.warning("House %d: dropping %d duplicate timestamps.", house_id, dup_mask.sum())
        df = df[~dup_mask]

    # Coerce all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    logger.info(
        "House %d: %d rows, %d columns, %s → %s",
        house_id,
        len(df),
        len(df.columns),
        df.index[0] if len(df) else "N/A",
        df.index[-1] if len(df) else "N/A",
    )
    return df
