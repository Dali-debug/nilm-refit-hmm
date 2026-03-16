#!/usr/bin/env python
"""
run_loho_train.py
=================
Leave-One-House-Out (LOHO) training.
For each test house, train HMM models on all remaining houses and save under
models/hmm/loho/test_house_XX/.

Usage
-----
    python scripts/run_loho_train.py [--houses 2 3 4 5 8] [--dev]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FULL_HOUSES, DEV_HOUSES, PROCESSED_DATA_DIR, MODELS_DIR, TARGET_APPLIANCES
)
from src.hmm_train import train_all_appliances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LOHO training of HMMs.")
    p.add_argument("--houses", nargs="+", type=int, default=None)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--n-iter", type=int, default=100)
    return p.parse_args()


def load_processed(processed_dir: Path, houses: list) -> dict:
    data = {}
    for hid in houses:
        path = processed_dir / f"house_{hid}.parquet"
        if not path.exists():
            logger.warning("Processed file not found: %s – run preprocess_all.py first.", path)
            continue
        data[hid] = pd.read_parquet(path)
        logger.info("Loaded house %d: %d rows", hid, len(data[hid]))
    return data


def main():
    args = parse_args()
    houses = args.houses or (DEV_HOUSES if args.dev else FULL_HOUSES)
    processed_dir = Path(args.processed_dir)

    if not processed_dir.exists():
        logger.error(
            "Processed data directory not found: %s\n"
            "Run scripts/preprocess_all.py first.",
            processed_dir.resolve(),
        )
        sys.exit(1)

    house_data = load_processed(processed_dir, houses)
    if len(house_data) < 2:
        logger.error("Need at least 2 processed houses for LOHO. Found: %d", len(house_data))
        sys.exit(1)

    available_houses = list(house_data.keys())
    models_dir = Path(args.models_dir)

    for test_house in available_houses:
        train_houses = [h for h in available_houses if h != test_house]
        logger.info("=== LOHO: test=%d, train=%s ===", test_house, train_houses)
        train_all_appliances(
            house_data=house_data,
            train_houses=train_houses,
            appliances=TARGET_APPLIANCES,
            models_dir=str(models_dir),
            test_house=test_house,
            n_iter=args.n_iter,
        )

    logger.info("LOHO training complete. Models saved under %s", models_dir.resolve())


if __name__ == "__main__":
    main()
