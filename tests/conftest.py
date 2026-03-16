"""
Pytest fixtures and synthetic data generator for NILM pipeline tests.
"""

import numpy as np
import pandas as pd
import pytest


def make_synthetic_house(
    n_minutes: int = 1440,  # 24 hours
    seed: int = 42,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Generate a synthetic REFIT-like house DataFrame (1-minute resolution).

    Columns: mains, kettle, microwave, fridge, washing_machine
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_minutes, freq="1min", tz="Europe/London")

    # Fridge: cyclic ~200W on, ~0W off, 20-min on / 10-min off
    fridge_cycle = np.zeros(n_minutes)
    t = 0
    on = True
    while t < n_minutes:
        dur = int(rng.integers(15, 25)) if on else int(rng.integers(8, 15))
        dur = min(dur, n_minutes - t)
        fridge_cycle[t: t + dur] = 200.0 + rng.normal(0, 5, dur) if on else rng.normal(0, 2, dur)
        on = not on
        t += dur
    fridge_cycle = np.clip(fridge_cycle, 0, None)

    # Kettle: ~3 short boil events per day
    kettle = np.zeros(n_minutes)
    for _ in range(3):
        start_t = int(rng.integers(0, n_minutes - 5))
        dur_t = int(rng.integers(2, 6))
        kettle[start_t: start_t + dur_t] = 2000.0 + rng.normal(0, 50, dur_t)
    kettle = np.clip(kettle, 0, None)

    # Microwave: ~2 short events
    microwave = np.zeros(n_minutes)
    for _ in range(2):
        start_t = int(rng.integers(0, n_minutes - 4))
        dur_t = int(rng.integers(1, 4))
        microwave[start_t: start_t + dur_t] = 900.0 + rng.normal(0, 30, dur_t)
    microwave = np.clip(microwave, 0, None)

    # Washing machine: 1 long cycle
    washing_machine = np.zeros(n_minutes)
    wm_start = int(rng.integers(100, 700))
    phases = [
        (40, 800),   # fill
        (60, 1500),  # wash
        (20, 400),   # rinse
        (30, 200),   # spin low
        (15, 1800),  # spin high
    ]
    t = wm_start
    for dur_t, power in phases:
        if t >= n_minutes:
            break
        end_t = min(t + dur_t, n_minutes)
        if end_t > t:
            washing_machine[t:end_t] = power + rng.normal(0, 20, end_t - t)
        t = end_t
    washing_machine = np.clip(washing_machine, 0, None)

    # Mains ≈ sum + background noise
    background = 50.0 + rng.normal(0, 10, n_minutes)
    mains = fridge_cycle + kettle + microwave + washing_machine + np.clip(background, 0, None)

    # Insert a few NaN values to test cleaning
    nan_idx = rng.integers(0, n_minutes, size=20)
    mains[nan_idx] = np.nan

    return pd.DataFrame(
        {
            "mains":           mains,
            "kettle":          kettle,
            "microwave":       microwave,
            "fridge":          fridge_cycle,
            "washing_machine": washing_machine,
        },
        index=index,
    )


@pytest.fixture(scope="session")
def synthetic_house():
    """Single synthetic house DataFrame (24h at 1min resolution)."""
    return make_synthetic_house(n_minutes=1440, seed=42)


@pytest.fixture(scope="session")
def synthetic_houses():
    """Dict of 3 synthetic house DataFrames for LOHO tests."""
    return {i: make_synthetic_house(n_minutes=1440, seed=i * 10) for i in range(3)}
