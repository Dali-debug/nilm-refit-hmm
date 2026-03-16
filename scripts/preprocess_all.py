#!/usr/bin/env python
"""
preprocess_all.py
=================
Load all selected houses from data/raw/, apply the appliance mapping,
clean and resample to 1-minute resolution, then save to
data/processed/1min/house_{id}.parquet.

Usage
-----
    python scripts/preprocess_all.py [--houses 2 3 4 5 8] [--dev]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FULL_HOUSES, DEV_HOUSES, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.io_refit import load_house
from src.mapping import standardize, HOUSE_MAP
from src.cleaning import clean_resample
from src.config import TARGET_APPLIANCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess REFIT houses.")
    p.add_argument("--houses", nargs="+", type=int, default=None,
                   help="House IDs to process.  Defaults to FULL_HOUSES.")
    p.add_argument("--dev", action="store_true",
                   help="Use DEV_HOUSES subset.")
    p.add_argument("--raw-dir", default=RAW_DATA_DIR)
    p.add_argument("--out-dir", default=PROCESSED_DATA_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    houses = args.houses or (DEV_HOUSES if args.dev else FULL_HOUSES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error(
            "Raw data directory not found: %s\n"
            "Place REFIT CSV files under data/raw/ (not committed to git).",
            raw_dir.resolve(),
        )
        sys.exit(1)

    summary_rows = []

    for hid in houses:
        if hid not in HOUSE_MAP:
            logger.warning("House %d not in HOUSE_MAP; skipping.", hid)
            continue
        try:
            logger.info("=== House %d ===", hid)
            raw = load_house(hid, raw_dir=str(raw_dir))
            std = standardize(raw, hid, appliances=TARGET_APPLIANCES)
            clean, stats = clean_resample(std, report=True)

            out_path = out_dir / f"house_{hid}.parquet"
            clean.to_parquet(out_path)
            logger.info("Saved %s (%d rows)", out_path, len(clean))

            row = {"house": hid, "rows": len(clean), **{k: v for k, v in (stats or {}).items()
                                                         if not isinstance(v, dict)}}
            summary_rows.append(row)
        except FileNotFoundError as exc:
            logger.error("House %d: %s", hid, exc)
        except Exception as exc:
            logger.error("House %d failed: %s", hid, exc, exc_info=True)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / "cleaning_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Cleaning summary saved to %s", summary_path)
    else:
        logger.warning("No houses processed successfully.")


if __name__ == "__main__":
    main()
