# NILM REFIT HMM

Non-Intrusive Load Monitoring on the [REFIT](https://www.refitsmarthomes.org/) dataset using Hidden Markov Models (HMM).

## Overview

This project implements a **supervised, per-appliance HMM** disaggregation pipeline with a **greedy iterative-additive** inference strategy, evaluated using **Leave-One-House-Out (LOHO)** cross-validation.

| Property | Value |
|----------|-------|
| Dataset | REFIT (21 UK houses) |
| Houses used | **5 houses: [2, 3, 4, 5, 8]** |
| Appliances | kettle, microwave, fridge, television, washing machine |
| Resampling | **1 minute** |
| Processed format | **CSV** (`data/processed/1min/`) |
| Evaluation | Leave-One-House-Out (LOHO) |

## Project Structure

```
nilm-refit-hmm/
├── configs/
│   └── config.yaml          ← all parameters (houses, channels, thresholds)
├── data/
│   ├── raw/                 ← place REFIT CSV files here (not tracked)
│   └── processed/
│       └── 1min/            ← generated CSV files (not tracked)
├── models/                  ← trained HMM pickles (not tracked)
├── notebooks/
│   ├── 01_overview.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_state_design.ipynb
│   ├── 04_train_hmm.ipynb
│   └── 05_evaluation.ipynb
├── results/                 ← KPI CSV files (not tracked)
├── scripts/
│   ├── preprocess_all.py    ← produces CSV files
│   ├── train_loho.py        ← LOHO HMM training
│   └── evaluate_loho.py     ← LOHO evaluation + KPIs
├── src/
│   ├── load_refit.py
│   ├── cleaning.py
│   ├── states.py
│   ├── hmm_model.py
│   ├── inference.py
│   └── metrics.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **No pyarrow or parquet dependency is needed.** All processed data is stored as plain CSV.

### 2. Download REFIT data

Download the REFIT Electrical Load Measurements dataset (CSV format) and place the house files in `data/raw/`:

```
data/raw/House_2.csv
data/raw/House_3.csv
data/raw/House_4.csv
data/raw/House_5.csv
data/raw/House_8.csv
```

Each CSV has the format:
```
Time, Aggregate, Appliance1, Appliance2, ..., Appliance9
```

### 3. Preprocess

```bash
python scripts/preprocess_all.py
```

This reads `data/raw/House_<N>.csv` for each of the 5 configured houses, applies:
- Negative value removal
- Outlier capping (P99.9 × 1.5)
- Resampling to **1 minute**
- Gap interpolation (max 5 consecutive minutes)

And writes **CSV** output to `data/processed/1min/house_<N>.csv`.

### 4. Train (LOHO)

```bash
python scripts/train_loho.py
```

Runs 5 LOHO folds (train on 4 houses, test on 1). Saves pickled HMMs to `models/loho_<test_house>/<appliance>.pkl`.

### 5. Evaluate

```bash
python scripts/evaluate_loho.py
```

Runs greedy iterative-additive disaggregation on each test house and computes KPIs:

| KPI | Description |
|-----|-------------|
| MAE | Mean Absolute Error (W) |
| RMSE | Root Mean Squared Error (W) |
| NRMSE | RMSE normalised by P95–P5 range |
| REE | Relative Energy Error |
| F1 | Binary ON/OFF F1-score |

Results are saved to `results/loho_<N>_metrics.csv` and `results/summary.csv`.

## Configuration

All parameters are in `configs/config.yaml`:

```yaml
houses: [2, 3, 4, 5, 8]       # 5-house LOHO experiment
resample_freq: "1min"           # 1-minute bins
data:
  raw_dir: data/raw
  processed_dir: data/processed/1min   # CSV output (no parquet)
```

To change the house selection, appliance list, or HMM hyper-parameters, edit `configs/config.yaml`.

## Appliance Channel Mapping

| House | kettle | microwave | fridge | television | washing_machine |
|-------|--------|-----------|--------|------------|-----------------|
| 2     | ch 8   | ch 5      | ch 1   | ch 4       | ch 2 |
| 3     | ch 9   | ch 8      | ch 2   | ch 7       | ch 6 |
| 4     | ch 9   | ch 8      | ch 1   | ch 7       | ch 4 |
| 5     | ch 8   | ch 7      | ch 1   | ch 6       | ch 3 |
| 8     | ch 9   | ch 8      | ch 1   | ch 7       | ch 4 |

## KPI Definitions

```python
# NRMSE (robust normalisation)
nrmse = rmse / (P95 - P5)   # of ground-truth power

# REE (relative energy error)
ree = |E_pred - E_true| / (E_true + ε)

# F1: binary ON/OFF, positive class = ON (P > on_threshold)
```

## Methodology

1. **Cleaning**: negative removal → outlier cap → 1-min resample → limited interpolation
2. **State labelling**: OFF state (P ≤ threshold) + KMeans clustering of ON values
3. **Training**: GaussianHMM with supervised initialisation, Baum-Welch refinement
4. **Inference**: greedy iterative-additive subtraction (kettle → microwave → fridge → tv → washing machine)
5. **Evaluation**: LOHO on 5 houses, per-appliance KPIs

## Dependencies

See `requirements.txt`. Main packages:
- `pandas`, `numpy` — data handling
- `hmmlearn` — GaussianHMM
- `scikit-learn` — KMeans, F1-score
- `matplotlib`, `seaborn` — plots
- `pyyaml` — config loading

**No pyarrow / parquet dependency.**
