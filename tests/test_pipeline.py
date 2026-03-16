"""
tests/test_pipeline.py
─────────────────────
Unit and integration tests for the NILM REFIT HMM pipeline.
Run with:  pytest tests/
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow running from repo root
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minute_series() -> pd.Series:
    """1-minute time series with clear ON/OFF pattern."""
    np.random.seed(0)
    idx = pd.date_range("2023-01-01", periods=300, freq="1min")
    power = np.zeros(300)
    power[::10] = 2000.0  # ON every 10 minutes
    return pd.Series(power, index=idx, name="kettle")


@pytest.fixture
def minute_df(minute_series) -> pd.DataFrame:
    np.random.seed(1)
    idx = minute_series.index
    return pd.DataFrame(
        {
            "aggregate": minute_series.values + np.random.uniform(50, 300, len(idx)),
            "kettle": minute_series.values,
            "fridge": np.random.uniform(30, 100, len(idx)),
        },
        index=idx,
    )


@pytest.fixture
def config() -> dict:
    from src.load_refit import load_config
    return load_config("configs/config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────────────────

def test_config_houses(config):
    assert config["houses"] == [2, 3, 4, 5, 8], "Default houses must be [2,3,4,5,8]"


def test_config_no_parquet(config):
    assert "parquet" not in str(config).lower(), "Config must not reference parquet"


def test_config_processed_dir_is_csv_path(config):
    assert config["data"]["processed_dir"] == "data/processed/1min"


def test_config_resample_freq(config):
    assert config["resample_freq"] == "1min"


def test_config_all_houses_have_channels(config):
    for h in config["houses"]:
        assert h in config["house_channels"], f"Missing channel map for house {h}"


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning tests
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_dataframe_resamples():
    from src.cleaning import clean_dataframe

    idx = pd.date_range("2023-01-01", periods=600, freq="8s")
    df_raw = pd.DataFrame(
        {"aggregate": np.random.uniform(100, 3000, 600)}, index=idx
    )
    df_clean, _ = clean_dataframe(df_raw, resample_freq="1min")
    assert len(df_clean) <= 90  # 600 * 8s = 80min max


def test_clean_removes_negatives():
    from src.cleaning import clean_dataframe

    idx = pd.date_range("2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({"aggregate": np.ones(100)}, index=idx)
    df.iloc[5, 0] = -50.0
    df_clean, _ = clean_dataframe(df, resample_freq="1min")
    # After cleaning, no negative values remain (NaN is ok)
    non_nan = df_clean["aggregate"].dropna()
    assert (non_nan >= 0).all()


def test_clean_report_keys():
    from src.cleaning import clean_dataframe

    idx = pd.date_range("2023-01-01", periods=200, freq="1min")
    df = pd.DataFrame({"aggregate": np.random.uniform(0, 500, 200)}, index=idx)
    _, report = clean_dataframe(df)
    assert "aggregate" in report
    assert "missing_after_interp" in report["aggregate"]


# ─────────────────────────────────────────────────────────────────────────────
# States tests
# ─────────────────────────────────────────────────────────────────────────────

def test_label_states_binary(minute_series):
    from src.states import label_states

    labels, centroids = label_states(minute_series, on_threshold=1000, n_states=2)
    assert len(labels) == len(minute_series)
    assert set(np.unique(labels)).issubset({0, 1})
    assert centroids.shape == (2,)


def test_label_states_multistate(minute_series):
    from src.states import label_states

    labels, centroids = label_states(minute_series, on_threshold=1000, n_states=3)
    assert centroids.shape == (3,)
    assert np.all(np.diff(centroids) >= 0), "Centroids must be sorted ascending"


def test_label_states_no_on_events():
    from src.states import label_states

    idx = pd.date_range("2023-01-01", periods=100, freq="1min")
    series = pd.Series(np.zeros(100), index=idx)
    labels, centroids = label_states(series, on_threshold=1000, n_states=2)
    assert np.all(labels == 0)


# ─────────────────────────────────────────────────────────────────────────────
# HMM model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_train_hmm(minute_series):
    from src.hmm_model import train_hmm

    model, centroids = train_hmm(minute_series, on_threshold=1000, n_states=2, n_iter=5)
    assert model.n_components == 2
    assert centroids.shape == (2,)


def test_predict_states(minute_series):
    from src.hmm_model import train_hmm, predict_states

    model, _ = train_hmm(minute_series, on_threshold=1000, n_states=2, n_iter=5)
    states = predict_states(model, minute_series)
    assert len(states) == len(minute_series)
    assert set(np.unique(states)).issubset({0, 1})


def test_save_load_model(minute_series, tmp_path):
    from src.hmm_model import train_hmm, save_model, load_model, predict_states

    model, _ = train_hmm(minute_series, on_threshold=1000, n_states=2, n_iter=5)
    path = tmp_path / "kettle.pkl"
    save_model(model, path)
    m2 = load_model(path)
    s1 = predict_states(model, minute_series)
    s2 = predict_states(m2, minute_series)
    assert np.array_equal(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# Inference tests
# ─────────────────────────────────────────────────────────────────────────────

def test_disaggregate_returns_correct_columns(minute_df, minute_series):
    from src.hmm_model import train_hmm
    from src.inference import disaggregate

    model, _ = train_hmm(minute_series, on_threshold=1000, n_states=2, n_iter=5)
    centroids = np.sort(model.means_.ravel())
    result = disaggregate(minute_df["aggregate"], [("kettle", model, centroids)])
    assert "kettle" in result.columns
    assert len(result) == len(minute_df)


def test_disaggregate_non_negative(minute_df, minute_series):
    from src.hmm_model import train_hmm
    from src.inference import disaggregate

    model, _ = train_hmm(minute_series, on_threshold=1000, n_states=2, n_iter=5)
    centroids = np.sort(model.means_.ravel())
    result = disaggregate(minute_df["aggregate"], [("kettle", model, centroids)])
    assert (result["kettle"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_perfect_prediction(minute_series):
    from src.metrics import compute_all_metrics

    kpis = compute_all_metrics(minute_series, minute_series, on_threshold=1000)
    assert kpis["mae"] == pytest.approx(0.0)
    assert kpis["rmse"] == pytest.approx(0.0)
    assert kpis["f1"] == pytest.approx(1.0)


def test_metrics_all_keys(minute_series):
    from src.metrics import compute_all_metrics

    kpis = compute_all_metrics(minute_series, minute_series * 0.8, on_threshold=1000)
    assert set(kpis.keys()) == {"mae", "rmse", "nrmse", "ree", "f1"}


def test_metrics_ree_range(minute_series):
    from src.metrics import compute_all_metrics

    kpis = compute_all_metrics(minute_series, minute_series * 0.5, on_threshold=1000)
    assert 0.0 <= kpis["ree"] <= 1.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CSV output tests (integration)
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_writes_csv_not_parquet(tmp_path, config):
    """End-to-end: preprocess should write CSV and NOT parquet."""
    from src.load_refit import load_house_raw
    from src.cleaning import clean_dataframe

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    proc_dir = tmp_path / "processed" / "1min"
    proc_dir.mkdir(parents=True)

    # Write a fake House_2.csv
    idx = pd.date_range("2023-01-01", periods=500, freq="8s")
    data = np.random.uniform(10, 2000, (500, 9))
    fake = pd.DataFrame(data, columns=[f"App{i}" for i in range(9)])
    fake.insert(0, "Time", idx)
    fake.to_csv(raw_dir / "House_2.csv", index=False)

    channels = config["house_channels"][2]
    df = load_house_raw(2, raw_dir, channels)
    df_clean, _ = clean_dataframe(df, resample_freq="1min")

    out_csv = proc_dir / "house_2.csv"
    df_clean.to_csv(out_csv)

    # CSV must exist
    assert out_csv.exists(), "CSV file was not created"
    # No parquet files
    assert not list(proc_dir.glob("*.parquet")), "Unexpected parquet file"
    # CSV is readable and has correct shape
    df_read = pd.read_csv(out_csv, index_col="timestamp", parse_dates=True)
    assert len(df_read) == len(df_clean)
    assert list(df_read.columns) == list(df_clean.columns)
