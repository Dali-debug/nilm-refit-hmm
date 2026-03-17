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
    prefer_unix_time: bool = False,
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
    prefer_unix_time : bool
        If True, always derive the index from the ``Unix`` epoch column,
        bypassing ``Time`` string parsing entirely.  This sidesteps DST
        ambiguity and is recommended for REFIT data spanning DST transitions.
        Default is False (use ``Time`` with automatic fallback to ``Unix``).

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

    def _time_from_unix() -> pd.Series:
        return pd.to_datetime(df["Unix"], unit="s", utc=True).dt.tz_convert(tz)

    if prefer_unix_time and "Unix" in df.columns:
        # Caller has explicitly requested Unix-epoch-based timestamps.
        logger.info("House %d: using Unix epoch column (prefer_unix_time=True).", house_id)
        df["Time"] = _time_from_unix()
    elif not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        # Time column could not be parsed at all; fall back to Unix.
        logger.warning(
            "House %d: Time column parse failed; falling back to Unix column.", house_id
        )
        df["Time"] = _time_from_unix()
    else:
        # Localise / convert
        if df["Time"].dt.tz is None:
            try:
                df["Time"] = df["Time"].dt.tz_localize(
                    tz,
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            except Exception as exc:
                # Some REFIT files contain DST-fallback duplicates that cannot
                # be inferred safely from local wall-clock timestamps.
                # pandas may raise AmbiguousTimeError (a ValueError subclass) or
                # a plain ValueError whose message mentions "ambiguous" or
                # "nonexistent" — check both forms so that all pandas versions
                # (and pytz/zoneinfo backends) are handled correctly.
                exc_name = exc.__class__.__name__
                exc_msg = str(exc).lower()
                is_dst_error = exc_name in (
                    "AmbiguousTimeError",
                    "NonExistentTimeError",
                ) or (
                    isinstance(exc, ValueError)
                    and any(kw in exc_msg for kw in ("ambiguous", "nonexistent"))
                )
                if is_dst_error and "Unix" in df.columns:
                    logger.warning(
                        "House %d: DST error during tz_localize (%s: %s). "
                        "Falling back to Unix epoch column.",
                        house_id,
                        exc_name,
                        exc,
                    )
                    df["Time"] = _time_from_unix()
                else:
                    raise
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
