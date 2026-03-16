#!/usr/bin/env python
"""
evaluate_loho.py
────────────────
LOHO evaluation: run greedy iterative-additive disaggregation on the test
house and compute KPIs (MAE, RMSE, NRMSE, REE, F1).

For each LOHO fold (test house h_test):
  1. Load trained models from  models/loho_<h_test>/
  2. Load processed CSV for h_test from  data/processed/1min/house_<h_test>.csv
  3. Disaggregate mains using the greedy algorithm
  4. Compute KPIs vs ground-truth sub-meters
  5. Save per-fold results to  results/loho_<h_test>_metrics.csv

A summary table (mean ± std across folds) is written to results/summary.csv.

Usage
-----
    python scripts/evaluate_loho.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.hmm_model import load_model
from src.inference import disaggregate
from src.load_refit import load_config
from src.metrics import compute_all_metrics, metrics_table
from src.states import label_states

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_processed_csv(processed_dir: Path, house_id: int) -> pd.DataFrame:
    path = processed_dir / f"house_{house_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {path}. Run preprocess_all.py first."
        )
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOHO evaluation.")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path(cfg["evaluation"]["models_dir"])
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    houses: list[int] = cfg["houses"]
    appliances: dict = cfg["appliances"]
    energy_eps: float = cfg["evaluation"]["energy_epsilon"]

    sorted_appliances = sorted(appliances.items(), key=lambda kv: kv[1]["order"])

    all_rows: list[dict] = []

    for test_house in houses:
        logger.info("═" * 60)
        logger.info("LOHO evaluation: test_house=%d", test_house)

        fold_model_dir = models_dir / f"loho_{test_house}"

        # Load processed test data
        try:
            df_test = load_processed_csv(processed_dir, test_house)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            continue

        if "aggregate" not in df_test.columns:
            logger.error("No 'aggregate' column in house %d CSV — skipping.", test_house)
            continue

        # Build appliance_models list for disaggregation
        appliance_models = []
        for app_name, app_cfg in sorted_appliances:
            model_path = fold_model_dir / f"{app_name}.pkl"
            if not model_path.exists():
                logger.warning("Model not found: %s — skipping '%s'.", model_path, app_name)
                continue
            model = load_model(model_path)
            # Re-derive centroids from the model means (sorted ascending)
            centroids = np.sort(model.means_.ravel())
            appliance_models.append((app_name, model, centroids))

        if not appliance_models:
            logger.error("No models loaded for house %d — skipping.", test_house)
            continue

        # Disaggregate
        predictions = disaggregate(df_test["aggregate"], appliance_models)

        # Compute KPIs per appliance
        fold_rows: list[dict] = []
        for app_name, app_cfg in sorted_appliances:
            if app_name not in predictions.columns:
                continue
            if app_name not in df_test.columns:
                logger.warning(
                    "Ground truth for '%s' not in house %d CSV.", app_name, test_house
                )
                continue

            kpis = compute_all_metrics(
                df_test[app_name],
                predictions[app_name],
                on_threshold=app_cfg["on_threshold"],
                energy_epsilon=energy_eps,
            )
            row = {"house": test_house, "appliance": app_name, **kpis}
            fold_rows.append(row)
            all_rows.append(row)
            logger.info(
                "  %-20s  MAE=%.1fW  RMSE=%.1fW  NRMSE=%.3f  REE=%.3f  F1=%.3f",
                app_name,
                kpis["mae"],
                kpis["rmse"],
                kpis["nrmse"],
                kpis["ree"],
                kpis["f1"],
            )

        # Save fold results
        fold_csv = results_dir / f"loho_{test_house}_metrics.csv"
        pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)
        logger.info("Fold results saved → %s", fold_csv)

    # Summary across all folds
    if all_rows:
        summary_df = metrics_table(all_rows)
        summary_path = results_dir / "summary.csv"
        summary_df.to_csv(summary_path)
        logger.info("Summary saved → %s", summary_path)

        # Print mean ± std per appliance
        numeric_cols = ["mae", "rmse", "nrmse", "ree", "f1"]
        mean_std = (
            summary_df[numeric_cols]
            .groupby(level="appliance")
            .agg(["mean", "std"])
        )
        logger.info("\n%s", mean_std.to_string())

    logger.info("═" * 60)
    logger.info("LOHO evaluation complete.  Results: %s", results_dir)


if __name__ == "__main__":
    main()
