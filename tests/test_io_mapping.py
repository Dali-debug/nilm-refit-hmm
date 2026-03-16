"""
Tests for io_refit and mapping modules using synthetic CSV data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mapping import standardize, HOUSE_MAP


def make_synthetic_csv(house_id: int, tmp_dir: Path, n_rows: int = 100) -> Path:
    """Create a synthetic REFIT-format CSV file."""
    rng = np.random.default_rng(house_id)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="8s")
    unix = (dates - pd.Timestamp("1970-01-01")).total_seconds().astype(int)

    data = {
        "Time":       dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Unix":       unix,
        "Aggregate":  rng.uniform(100, 3000, n_rows),
    }
    for i in range(1, 10):
        data[f"Appliance{i}"] = rng.uniform(0, 500, n_rows)

    df = pd.DataFrame(data)
    csv_path = tmp_dir / f"House_{house_id}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestIORefit:
    def test_load_house(self):
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_synthetic_csv(2, tmp_path, n_rows=50)
            df = load_house(2, raw_dir=str(tmp_path))
            assert len(df) == 50
            assert "Aggregate" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_missing_raw_dir_raises(self):
        from src.io_refit import load_house
        with pytest.raises(FileNotFoundError, match="Raw data directory"):
            load_house(2, raw_dir="/nonexistent/path")

    def test_missing_csv_raises(self):
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="House CSV not found"):
                load_house(99, raw_dir=tmp)

    def test_duplicate_timestamps_handled(self):
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = make_synthetic_csv(2, tmp_path, n_rows=50)
            # Add duplicate row
            df = pd.read_csv(csv_path)
            df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
            df.to_csv(csv_path, index=False)
            loaded = load_house(2, raw_dir=str(tmp_path))
            assert loaded.index.is_unique


class TestMapping:
    def test_house_map_has_required_houses(self):
        for h in [2, 3, 4, 5, 8, 9, 15, 20]:
            assert h in HOUSE_MAP, f"House {h} missing from HOUSE_MAP"

    def test_standardize_output_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            from src.io_refit import load_house
            tmp_path = Path(tmp)
            make_synthetic_csv(2, tmp_path, n_rows=50)
            raw = load_house(2, raw_dir=str(tmp_path))
            std = standardize(raw, house_id=2)
            for col in ["mains", "kettle", "microwave", "fridge", "washing_machine"]:
                assert col in std.columns

    def test_standardize_house4_sums_channels(self):
        """House 4 fridge = Appliance1 + Appliance3; check sum."""
        with tempfile.TemporaryDirectory() as tmp:
            from src.io_refit import load_house
            tmp_path = Path(tmp)
            make_synthetic_csv(4, tmp_path, n_rows=50)
            raw = load_house(4, raw_dir=str(tmp_path))
            std = standardize(raw, house_id=4)
            expected = (raw["Appliance1"] + raw["Appliance3"]).values
            np.testing.assert_allclose(std["fridge"].values, expected, rtol=1e-5)

    def test_standardize_invalid_house_raises(self):
        df = pd.DataFrame({"Aggregate": [1, 2]})
        with pytest.raises(ValueError, match="not in HOUSE_MAP"):
            standardize(df, house_id=999)
