#!/usr/bin/env python
"""
train_loho.py
─────────────
Leave-One-House-Out (LOHO) training of per-appliance GaussianHMMs.

For each test house h_test:
  - Train on all other houses (h_train = houses \\ {h_test})
  - Concatenate training data for each appliance across h_train houses
  - Fit one GaussianHMM per appliance
  - Save models to  models/loho_<h_test>/<appliance>.pkl

Usage
-----
    python scripts/train_loho.py [--config configs/config.yaml]

Input
-----
    data/processed/1min/house_<N>.csv   (CSV, produced by preprocess_all.py)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.hmm_model import save_model, train_hmm
from src.load_refit import load_config

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
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOHO HMM training.")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path(cfg["evaluation"]["models_dir"])
    houses: list[int] = cfg["houses"]
    appliances: dict = cfg["appliances"]
    hmm_cfg: dict = cfg["hmm"]

    # Sort appliances by greedy order
    sorted_appliances = sorted(appliances.items(), key=lambda kv: kv[1]["order"])

    for test_house in houses:
        train_houses = [h for h in houses if h != test_house]
        logger.info("═" * 60)
        logger.info(
            "LOHO fold: test_house=%d  train_houses=%s", test_house, train_houses
        )

        fold_model_dir = models_dir / f"loho_{test_house}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        # Load training data for all train houses
        house_dfs: dict[int, pd.DataFrame] = {}
        for h in train_houses:
            try:
                house_dfs[h] = load_processed_csv(processed_dir, h)
            except FileNotFoundError as exc:
                logger.error("%s", exc)

        if not house_dfs:
            logger.error("No training data available — skipping fold.")
            continue

        for app_name, app_cfg in sorted_appliances:
            logger.info("  Training HMM for '%s' …", app_name)

            # Concatenate appliance series across train houses
            series_list = []
            for h, df in house_dfs.items():
                if app_name in df.columns:
                    series_list.append(df[app_name].dropna())
                else:
                    logger.warning(
                        "  House %d has no column '%s' — skipping.", h, app_name
                    )

            if not series_list:
                logger.warning(
                    "  No training data for '%s' — skipping.", app_name
                )
                continue

            combined = pd.concat(series_list)

            try:
                model, _centroids = train_hmm(
                    combined,
                    on_threshold=app_cfg["on_threshold"],
                    n_states=app_cfg["n_states"],
                    covariance_type=hmm_cfg["covariance_type"],
                    n_iter=hmm_cfg["n_iter"],
                    tol=hmm_cfg["tol"],
                    random_state=hmm_cfg["random_state"],
                )
                save_model(model, fold_model_dir / f"{app_name}.pkl")
            except Exception as exc:
                logger.error(
                    "  Failed to train HMM for '%s': %s", app_name, exc
                )

    logger.info("═" * 60)
    logger.info("LOHO training complete.  Models: %s", models_dir)


if __name__ == "__main__":
    main()
