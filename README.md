# NILM-REFIT-HMM

Non-Intrusive Load Monitoring (NILM) using Hidden Markov Models on the [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) dataset.

## Overview

This project implements a **Gaussian HMM** disaggregation pipeline with a **Leave-One-House-Out (LOHO)** cross-validation strategy. It targets four appliances — kettle, microwave, fridge, and washing machine — across 8 REFIT households.

### Key features

- **Greedy additive disaggregation**: sequentially subtracts each appliance from the residual mains signal using Viterbi decoding.
- **Data-driven state initialisation**: KMeans clustering on submeter data initialises HMM means/covariances for faster convergence.
- **Configurable pipeline**: all hyperparameters live in `src/config.py`.
- **Comprehensive evaluation**: MAE, RMSE, NRMSE, REE, and F1 (ON/OFF).

## Project structure

```
nilm-refit-hmm/
├── data/
│   ├── raw/                   # REFIT CSVs (not committed – place here manually)
│   └── processed/1min/        # Output of preprocess_all.py
├── models/hmm/loho/           # Trained HMM models (joblib)
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_data_overview.ipynb
│   ├── 02_cleaning_and_resampling.ipynb
│   ├── 03_appliance_mapping_check.ipynb
│   ├── 04_state_design.ipynb
│   ├── 05_train_hmm_cross_house.ipynb
│   ├── 06_greedy_inference_and_metrics.ipynb
│   └── 07_results_tables_and_plots.ipynb
├── results/loho/              # Predictions and metrics CSVs
├── scripts/
│   ├── preprocess_all.py      # Step 1: load → map → clean → parquet
│   ├── run_loho_train.py      # Step 2: LOHO HMM training
│   ├── run_loho_infer.py      # Step 3: inference + evaluation
│   └── export_results.py      # Step 4: aggregate summary tables
├── src/
│   ├── config.py              # Central configuration
│   ├── io_refit.py            # CSV loader
│   ├── mapping.py             # House → appliance column mapping
│   ├── cleaning.py            # Resampling and outlier removal
│   ├── utils.py               # Segment utilities
│   ├── states.py              # HMM state initialisation
│   ├── hmm_train.py           # GaussianHMM training
│   ├── inference_greedy.py    # Greedy disaggregation
│   ├── metrics.py             # Evaluation metrics
│   └── plots.py               # Visualisation helpers
├── tests/
│   ├── conftest.py            # Synthetic data fixtures
│   ├── test_pipeline.py       # End-to-end pipeline tests
│   └── test_io_mapping.py     # I/O and mapping tests
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain REFIT data

Download the REFIT Electrical Load Measurements dataset from the [Strathclyde data portal](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) and place the `House_*.csv` files in `data/raw/`. These files are excluded from version control (`.gitignore`).

## Usage

### Run the full pipeline

```bash
# Step 1: preprocess raw CSVs → parquet
python scripts/preprocess_all.py

# Step 2: LOHO training (all 8 houses)
python scripts/run_loho_train.py

# Step 3: inference + metrics
python scripts/run_loho_infer.py

# Step 4: aggregate results
python scripts/export_results.py
```

### Development mode (houses 2 & 3 only)

```bash
python scripts/preprocess_all.py --dev
python scripts/run_loho_train.py --dev
python scripts/run_loho_infer.py --dev
```

### Custom house selection

```bash
python scripts/preprocess_all.py --houses 2 3 5 8
```

## Configuration

Edit `src/config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FULL_HOUSES` | `[2,3,4,5,8,9,15,20]` | Houses used in full experiments |
| `DEV_HOUSES` | `[2,3]` | Quick development subset |
| `TARGET_APPLIANCES` | see config | Appliances to disaggregate |
| `N_STATES` | per appliance | Number of HMM states |
| `RESAMPLE_RULE` | `"1min"` | Target temporal resolution |
| `INTERP_LIMIT` | `5` | Max NaN gap to interpolate |
| `POWER_CAP` | per appliance | Outlier clipping threshold (W) |

## Running tests

```bash
python -m pytest tests/ -v
```

Tests use **synthetic data only** — no REFIT files are required.

## Appliance-to-column mapping

| House | Kettle | Microwave | Fridge | Washing Machine |
|-------|--------|-----------|--------|-----------------|
| 2 | Appliance8 | Appliance5 | Appliance1 | Appliance2 |
| 3 | Appliance9 | Appliance8 | Appliance2 | Appliance6 |
| 4 | Appliance9 | Appliance8 | App1+App3 | App4+App5 |
| 5 | Appliance8 | Appliance7 | Appliance1 | Appliance3 |
| 8 | Appliance9 | Appliance8 | Appliance1 | Appliance4 |
| 9 | Appliance7 | Appliance6 | Appliance1 | Appliance3 |
| 15 | Appliance8 | Appliance7 | Appliance1 | Appliance3 |
| 20 | Appliance9 | Appliance8 | Appliance1 | Appliance4 |

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (Watts) |
| **RMSE** | Root Mean Squared Error (Watts) |
| **NRMSE** | RMSE / (P95 − P5) of ground truth |
| **REE** | Relative Energy Error: `|E_pred − E_true| / E_true` |
| **F1** | Binary ON/OFF F1-score (threshold = 10 W) |

## Citation

If you use this code, please cite the REFIT dataset:

> Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Scientific Data*, 4, 160122.
