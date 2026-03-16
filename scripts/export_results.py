#!/usr/bin/env python
"""
export_results.py
=================
Aggregate per-house metrics into summary tables.
Reads results/loho/metrics.csv and outputs summary statistics.

Usage
-----
    python scripts/export_results.py [--results-dir results/loho]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate LOHO results.")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics_path = results_dir / "metrics.csv"

    if not metrics_path.exists():
        logger.error("metrics.csv not found at %s – run run_loho_infer.py first.", metrics_path)
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    logger.info("Loaded metrics: %d rows", len(df))

    metric_cols = [c for c in df.columns if c not in ("house", "appliance")]

    # Per-appliance summary (mean ± std across houses)
    agg = df.groupby("appliance")[metric_cols].agg(["mean", "std"]).round(4)
    print("\n=== Per-appliance summary (mean ± std across houses) ===")
    print(agg.to_string())

    summary_path = results_dir / "summary_per_appliance.csv"
    agg.to_csv(summary_path)
    logger.info("Per-appliance summary saved to %s", summary_path)

    # Per-house summary (mean across appliances)
    if "house" in df.columns:
        per_house = df.groupby("house")[metric_cols].mean().round(4)
        print("\n=== Per-house summary (mean across appliances) ===")
        print(per_house.to_string())
        per_house_path = results_dir / "summary_per_house.csv"
        per_house.to_csv(per_house_path)
        logger.info("Per-house summary saved to %s", per_house_path)

    # Overall mean
    overall = df[metric_cols].mean().round(4)
    print("\n=== Overall mean metrics ===")
    print(overall.to_string())


if __name__ == "__main__":
    main()
