"""
Central configuration for the NILM-REFIT-HMM project.
All tunable parameters live here so scripts/notebooks stay clean.
"""

# ── Houses ────────────────────────────────────────────────────────────────────
# Development subset (fast iteration)
DEV_HOUSES = [2, 3]

# Full experimental set – houses with reliable submeter data for target appliances
FULL_HOUSES = [2, 3, 4, 5, 8, 9, 15, 20]

# ── Target appliances ─────────────────────────────────────────────────────────
TARGET_APPLIANCES = ["kettle", "microwave", "fridge", "washing_machine"]

# ── ON/OFF thresholds (Watts) – used for F1 evaluation ───────────────────────
ON_THRESHOLD = {
    "kettle":          10.0,
    "microwave":       10.0,
    "fridge":          10.0,
    "washing_machine": 10.0,
}

# ── HMM states per appliance ──────────────────────────────────────────────────
N_STATES = {
    "kettle":          2,   # OFF / ON
    "microwave":       2,   # OFF / ON
    "fridge":          3,   # OFF / COMPRESSOR / HIGH
    "washing_machine": 4,   # OFF + 3 power phases
}

# ── Greedy inference order (simplest appliances first) ───────────────────────
INFERENCE_ORDER = ["kettle", "microwave", "fridge", "washing_machine"]

# ── Pre-processing ────────────────────────────────────────────────────────────
RESAMPLE_RULE    = "1min"   # target temporal resolution
INTERP_LIMIT     = 5        # max consecutive NaN minutes to interpolate

# Per-appliance power cap for outlier removal (Watts)
POWER_CAP = {
    "mains":           15_000,
    "kettle":           4_000,
    "microwave":        3_000,
    "fridge":           1_000,
    "washing_machine":  4_000,
}

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Timestamp handling ────────────────────────────────────────────────────────
# Set to True to always derive the index from the Unix epoch column,
# sidestepping DST wall-clock ambiguity for REFIT data.
PREFER_UNIX_TIME = False

# ── Paths (relative to project root) ─────────────────────────────────────────
RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed/1min"
MODELS_DIR         = "models/hmm/loho"
RESULTS_DIR        = "results/loho"
