#!/usr/bin/env python

import pandas as pd
import os
import glob
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--iteration",
    type=int,
    help="The iterator of the specific combined .csv to run",
)
args = parser.parse_args()

dirTextFile = (
    "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/prod_directories.txt"
)
input_folder = Path(dirTextFile).read_text().splitlines()[args.iteration]

logger.info(f"Processing .csv's in {input_folder}...")

output_file = os.path.join(input_folder, "combined.csv")

files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
first_file = True

logger.info(f"All files in {input_folder}:")
for f in files:
    print(f)

for filename in files:
    file_path = os.path.join(input_folder, filename)
    logger.info(f"Processing {filename}...")

    # Read the file in chunks
    chunk = pd.read_csv(file_path)
    if first_file:
        # 1. Establish the Master Header from the very first chunk
        master_columns = chunk.columns.tolist()
        chunk.to_csv(output_file, index=False, mode="w")
        first_file = False
    else:
        # 2. Reorder this chunk to match the Master Header
        # Any missing columns will be filled with NaN
        chunk = chunk.reindex(columns=master_columns)
        chunk.to_csv(output_file, index=False, mode="a", header=False)
    del chunk

logger.info(f"Finished! Combined data saved to {output_file}")
