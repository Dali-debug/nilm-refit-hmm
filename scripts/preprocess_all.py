#!/usr/bin/env python
"""
preprocess_all.py
─────────────────
Read raw REFIT CSVs for the configured houses, apply the cleaning pipeline,
and write processed CSV files to data/processed/1min/.

Usage
-----
    python scripts/preprocess_all.py [--config configs/config.yaml]

Output
------
    data/processed/1min/house_<N>.csv   (one file per house)

No pyarrow / parquet dependency is required.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_dataframe
from src.load_refit import load_config, load_house_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess REFIT houses to CSV.")
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    houses: list[int] = cfg["houses"]
    house_channels: dict = cfg["house_channels"]
    clean_cfg: dict = cfg["cleaning"]
    resample_freq: str = cfg["resample_freq"]

    logger.info("Houses to process: %s", houses)

    for house_id in houses:
        logger.info("═" * 60)
        logger.info("Processing house %d …", house_id)

        channels = house_channels.get(house_id)
        if channels is None:
            logger.warning("No channel mapping for house %d — skipping.", house_id)
            continue

        try:
            df_raw = load_house_raw(house_id, raw_dir, channels)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            continue

        df_clean, report = clean_dataframe(
            df_raw,
            resample_freq=resample_freq,
            outlier_quantile=clean_cfg["outlier_quantile"],
            interp_method=clean_cfg["interp_method"],
            interp_limit=clean_cfg["interp_limit"],
            max_missing_ratio=clean_cfg["max_missing_ratio"],
        )

        # Log cleaning report
        for col, stats in report.items():
            flag = "⚠" if stats.get("too_many_missing") else " "
            logger.info(
                "  %s %-20s  missing_before=%.1f%%  missing_after=%.1f%%",
                flag,
                col,
                stats.get("missing_before_interp", 0) * 100,
                stats.get("missing_after_interp", 0) * 100,
            )

        # Write CSV — no pyarrow required
        out_path = processed_dir / f"house_{house_id}.csv"
        df_clean.to_csv(out_path)
        logger.info("Saved → %s  (%d rows × %d cols)", out_path, *df_clean.shape)

    logger.info("═" * 60)
    logger.info("Preprocessing complete.  Output: %s", processed_dir)


if __name__ == "__main__":
    main()
