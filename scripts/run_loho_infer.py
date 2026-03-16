#!/usr/bin/env python
"""
run_loho_infer.py
=================
Run greedy NILM inference for each test house using its LOHO models.
Saves predictions parquet and metrics CSV under results/loho/.

Usage
-----
    python scripts/run_loho_infer.py [--houses 2 3 4 5 8] [--dev]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FULL_HOUSES, DEV_HOUSES, PROCESSED_DATA_DIR, MODELS_DIR,
    RESULTS_DIR, TARGET_APPLIANCES, INFERENCE_ORDER
)
from src.hmm_train import load_model
from src.inference_greedy import greedy_disaggregate
from src.metrics import evaluate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LOHO inference + evaluation.")
    p.add_argument("--houses", nargs="+", type=int, default=None)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    houses = args.houses or (DEV_HOUSES if args.dev else FULL_HOUSES)

    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for test_house in houses:
        model_subdir = models_dir / f"test_house_{test_house:02d}"
        if not model_subdir.exists():
            logger.warning("No models for test_house %d at %s – skipping.", test_house, model_subdir)
            continue

        data_path = processed_dir / f"house_{test_house}.parquet"
        if not data_path.exists():
            logger.warning("No processed data for house %d – skipping.", test_house)
            continue

        df = pd.read_parquet(data_path)
        logger.info("=== Inference: house %d (%d rows) ===", test_house, len(df))

        # Load models
        models = {}
        for app in TARGET_APPLIANCES:
            mdl_path = model_subdir / f"{app}.pkl"
            if mdl_path.exists():
                models[app] = load_model(str(mdl_path))
            else:
                logger.warning("Model not found: %s", mdl_path)

        if not models:
            logger.error("No models loaded for house %d; skipping.", test_house)
            continue

        # Disaggregate
        preds = greedy_disaggregate(df["mains"], models, order=INFERENCE_ORDER)

        # Save predictions
        pred_path = results_dir / f"house_{test_house}_predictions.parquet"
        preds.to_parquet(pred_path)
        logger.info("Predictions saved to %s", pred_path)

        # Evaluate
        apps_to_eval = [a for a in TARGET_APPLIANCES if a in df.columns and a in preds.columns]
        metrics_df = evaluate_all(df[apps_to_eval], preds[apps_to_eval],
                                  appliances=apps_to_eval, house_id=test_house)
        all_metrics.append(metrics_df)
        logger.info("\n%s", metrics_df.to_string(index=False))

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        metrics_path = results_dir / "metrics.csv"
        combined.to_csv(metrics_path, index=False)
        logger.info("All metrics saved to %s", metrics_path)
    else:
        logger.warning("No metrics computed.")


if __name__ == "__main__":
    main()
