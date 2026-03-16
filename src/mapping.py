"""
REFIT house-to-appliance column mapping.

For each house we define which raw Appliance columns (or sums thereof)
correspond to the standard target appliances.

Standard output columns: mains, kettle, microwave, fridge, washing_machine
"""

from typing import Dict, List, Optional, Union
import pandas as pd

# ── House map ─────────────────────────────────────────────────────────────────
# Format:
#   house_id -> {standard_name: column_name_or_list_of_columns_to_sum}
# Column names correspond to the raw CSV after io_refit.load_house().
# "Aggregate" is always available as mains.

HOUSE_MAP: Dict[int, Dict[str, Union[str, List[str]]]] = {
    2: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge-Freezer
        "washing_machine": "Appliance2",   # Washing Machine
        "microwave":       "Appliance5",   # Microwave
        "kettle":          "Appliance8",   # Kettle
    },
    3: {
        "mains":           "Aggregate",
        "fridge":          "Appliance2",   # Fridge-Freezer
        "washing_machine": "Appliance6",   # Washing Machine
        "microwave":       "Appliance8",   # Microwave
        "kettle":          "Appliance9",   # Kettle
    },
    4: {
        "mains":           "Aggregate",
        # House 4 has two fridge units (Fridge + Fridge-Freezer) → sum
        "fridge":          ["Appliance1", "Appliance3"],  # Fridge + Fridge-Freezer
        # Two washing machines → sum
        "washing_machine": ["Appliance4", "Appliance5"],
        "microwave":       "Appliance8",
        "kettle":          "Appliance9",
    },
    5: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge-Freezer
        "washing_machine": "Appliance3",   # Washing Machine
        "microwave":       "Appliance7",   # Combination Microwave
        "kettle":          "Appliance8",   # Kettle
    },
    8: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge
        "washing_machine": "Appliance4",   # Washing Machine
        "microwave":       "Appliance8",   # Microwave
        "kettle":          "Appliance9",   # Kettle
    },
    9: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge-Freezer
        "washing_machine": "Appliance3",   # Washing Machine
        "microwave":       "Appliance6",   # Microwave
        "kettle":          "Appliance7",   # Kettle
    },
    15: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge-Freezer
        "washing_machine": "Appliance3",   # Washing Machine
        "microwave":       "Appliance7",   # Microwave
        "kettle":          "Appliance8",   # Kettle
    },
    20: {
        "mains":           "Aggregate",
        "fridge":          "Appliance1",   # Fridge
        "washing_machine": "Appliance4",   # Washing Machine
        "microwave":       "Appliance8",   # Microwave
        "kettle":          "Appliance9",   # Kettle
    },
}


def standardize(
    df: pd.DataFrame,
    house_id: int,
    appliances: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Map raw REFIT columns to standardised appliance columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from io_refit.load_house().
    house_id : int
        House number; must exist in HOUSE_MAP.
    appliances : list[str] or None
        Subset of standard names to include.  Defaults to all mapped.

    Returns
    -------
    pd.DataFrame
        Columns: mains + requested appliances (NaN where not mapped).
    """
    if house_id not in HOUSE_MAP:
        raise ValueError(
            f"House {house_id} is not in HOUSE_MAP. "
            f"Available: {sorted(HOUSE_MAP.keys())}"
        )

    mapping = HOUSE_MAP[house_id]
    all_names = list(mapping.keys())
    if appliances is not None:
        # always keep mains
        target_names = ["mains"] + [a for a in appliances if a != "mains"]
    else:
        target_names = all_names

    out: Dict[str, pd.Series] = {}
    for name in target_names:
        if name not in mapping:
            out[name] = pd.Series(float("nan"), index=df.index, name=name)
            continue
        src = mapping[name]
        if isinstance(src, list):
            # Sum multiple channels; treat NaN as 0 only when at least one channel
            # is non-NaN (fill_value=0 in add)
            series = df[src[0]].copy().rename(name)
            for col in src[1:]:
                series = series.add(df[col], fill_value=0)
            out[name] = series
        else:
            out[name] = df[src].rename(name)

    return pd.DataFrame(out)
