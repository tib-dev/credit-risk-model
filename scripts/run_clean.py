#!/usr/bin/env python
"""
Run Data Cleaning using Settings object.
"""

import argparse
import logging
from pathlib import Path

from credit_risk_model.core.settings import settings
from credit_risk_model.data.data_cleaning import DataCleaning
from credit_risk_model.data.load_data import DataLoader

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Data Cleaning")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("transactions.csv"),
        help="Raw CSV filename located in data/raw",
    )
    args = parser.parse_args()

    raw_path = settings.paths.data["raw_dir"] / args.file
    interim_path = (
        settings.paths.data["interim_dir"]
        / f"cleaned_{args.file.stem}.csv"
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    logger.info("Loading raw data from %s", raw_path)

    loader = DataLoader(filepath=raw_path, file_type="csv")
    raw_df = loader.load()

    cleaner = DataCleaning()
    clean_df = cleaner.clean(raw_df)

    clean_df.to_csv(interim_path, index=False)
    logger.info("Cleaned data saved to %s", interim_path)


if __name__ == "__main__":
    main()
