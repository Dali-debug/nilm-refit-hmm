"""
load_refit.py
─────────────
Utilities to load raw REFIT house CSV files.

REFIT CSV format (House_<N>.csv):
  Time, Aggregate, Appliance1, Appliance2, ..., Appliance9

The channel mapping (which column → which appliance) is defined in
configs/config.yaml under ``house_channels``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def load_house_raw(
    house_id: int,
    raw_dir: str | Path,
    house_channels: dict,
) -> pd.DataFrame:
    """Load a single REFIT house CSV and return a labelled DataFrame.

    Parameters
    ----------
    house_id:
        Integer house number (e.g. 2 for House_2.csv).
    raw_dir:
        Directory containing House_<N>.csv files.
    house_channels:
        Channel mapping for this house, e.g.
        ``{"aggregate": 0, "kettle": 8, "fridge": 1, ...}``.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (UTC-naive).
        Columns: named appliances + "aggregate".
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / f"House_{house_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {csv_path}. "
            "Download REFIT and place CSV files in the raw data directory."
        )

    logger.info("Loading house %d from %s …", house_id, csv_path)

    # REFIT CSVs use Unix timestamps or datetime strings — detect automatically
    df_raw = pd.read_csv(csv_path, low_memory=False)

    # Identify the time column (first column)
    time_col = df_raw.columns[0]
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    df_raw = df_raw.set_index(time_col).sort_index()
    df_raw.index.name = "timestamp"

    # Select and rename columns according to channel mapping
    # Column order in CSV: Aggregate=col1, Appliance1=col2, …
    all_cols = list(df_raw.columns)  # 0-indexed after dropping time column

    selected: dict[str, pd.Series] = {}
    for label, ch_idx in house_channels.items():
        if ch_idx < len(all_cols):
            selected[label] = pd.to_numeric(df_raw.iloc[:, ch_idx], errors="coerce")
        else:
            logger.warning(
                "House %d: channel %d for '%s' out of range — skipping.",
                house_id,
                ch_idx,
                label,
            )

    return pd.DataFrame(selected)
