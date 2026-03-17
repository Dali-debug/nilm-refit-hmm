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


def make_dst_ambiguous_csv(house_id: int, tmp_dir: Path) -> Path:
    """
    Create a REFIT-format CSV whose Time column spans the UK DST fall-back
    (2014-10-26 01:00 local time occurs twice), making tz_localize('infer')
    raise an AmbiguousTimeError / ValueError.  The Unix column is unambiguous.
    """
    # Build a sequence of UTC timestamps around the DST boundary.
    # UK clocks fall back from BST→GMT at 01:00 BST (= 00:00 UTC) on
    # 2014-10-26, so wall-clock 01:00–01:59 Europe/London repeats when
    # localised from naive strings.  Unix column remains unambiguous.
    start_utc = pd.Timestamp("2014-10-26 00:50:00", tz="UTC")
    utc_times = pd.date_range(start_utc, periods=30, freq="1min", tz="UTC")

    # Wall-clock (naive) strings – these are intentionally ambiguous
    local_strings = utc_times.tz_convert("Europe/London").strftime("%Y-%m-%d %H:%M:%S")
    unix_vals = utc_times.astype("int64") // 10 ** 9  # seconds since epoch

    rng = np.random.default_rng(house_id)
    data = {
        "Time": local_strings,
        "Unix": unix_vals,
        "Aggregate": rng.uniform(100, 3000, len(utc_times)),
    }
    for i in range(1, 10):
        data[f"Appliance{i}"] = rng.uniform(0, 500, len(utc_times))

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

    def test_dst_ambiguity_fallback_to_unix(self):
        """
        When Time column contains DST-ambiguous wall-clock strings that
        tz_localize cannot infer, load_house must fall back to the Unix
        column and succeed (no exception raised).
        """
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_dst_ambiguous_csv(2, tmp_path)
            # Should NOT raise — automatic fallback to Unix expected
            df = load_house(2, raw_dir=str(tmp_path), tz="Europe/London")
            assert len(df) > 0
            assert df.index.tz is not None
            assert df.index.is_unique

    def test_prefer_unix_time_skips_localize(self):
        """
        With prefer_unix_time=True the function must not attempt tz_localize
        and must produce a valid tz-aware index regardless of Time column content.
        """
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Even a file with DST-ambiguous Time strings must succeed.
            make_dst_ambiguous_csv(2, tmp_path)
            df = load_house(2, raw_dir=str(tmp_path), prefer_unix_time=True)
            assert len(df) > 0
            assert df.index.tz is not None

    def test_prefer_unix_time_normal_file(self):
        """prefer_unix_time=True works on an ordinary (non-ambiguous) CSV."""
        from src.io_refit import load_house
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            make_synthetic_csv(2, tmp_path, n_rows=50)
            df = load_house(2, raw_dir=str(tmp_path), prefer_unix_time=True)
            assert len(df) == 50
            assert df.index.tz is not None


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
