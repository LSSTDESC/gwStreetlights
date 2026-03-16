import argparse
import logging
import os
import itertools
import pandas as pd
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [ %(levelname)s ] - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def get_csv_path(manifest_path, line_number):
    """Efficiently grabs a filename from a specific line in a text file."""
    try:
        with open(manifest_path, 'r') as f:
            # line_number - 1 because islice is 0-indexed
            line = next(itertools.islice(f, line_number - 1, line_number), None)
            return line.strip() if line else None
    except FileNotFoundError:
        logging.error(f"Manifest file not found: {manifest_path}")
        return None

def add_column_to_csv(input_csv, target_col, new_col_name,config_year):

    redFuncKwargs= {"year":config_year}
    if new_col_name=="redshift_spectro":
        redFunc = ut.trueZ_to_specZ
    elif new_col_name=="redshift_modeled":
        redFunc = ut.trueZ_to_photoZ
        redFuncKwargs["modeled"] = True
    else:
        raise ValueError(f"Supplied column name {new_col_name}, only redshift_spectro and redshift_modeled are supported.")
    
    temp_csv = f"/pscratch/sd/s/seanmacb/{input_csv}.tmp"
    chunk_size = 1000000  # Adjust based on your available RAM
    
    logging.info(f"Processing file: {input_csv}")
    
    try:
        # 1. Initialize the reader
        # We use usecols to only pull the column we need for math + 
        # whatever columns we need to keep the file intact.
        # However, to maintain the FULL original file, we read all columns.
        reader = pd.read_csv(input_csv, chunksize=chunk_size)
        
        for i, chunk in enumerate(reader):
            # 2. Compute the new column (this creates it in the DataFrame)
            # This works even if new_col_name doesn't exist yet
            chunk[new_col_name] = redFunc(chunk[target_col],**redFuncKwargs)
            
            # 3. Write to temp file
            # Header is only written for the very first chunk (i == 0)
            is_first_chunk = (i == 0)
            chunk.to_csv(
                temp_csv, 
                mode='a' if not is_first_chunk else 'w', 
                index=False, 
                header=is_first_chunk
            )
            
            if i % 10 == 0:
                logging.info(f"Processed row {i * chunk_size:,}...")

        # 4. Atomic Swap: Replace old file with updated file
        os.replace(temp_csv, input_csv)
        logging.info(f"Successfully added '{new_col_name}' to {input_csv}")

    except Exception as e:
        logging.error(f"Failed during CSV processing: {e}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Add a calculated column to a specific CSV from a manifest.")
    parser.add_argument("line_number", type=int, help="The 1-based line number in the manifest file.")
    parser.add_argument("--manifest", default="manifest.txt", help="Path to your text file of CSV paths.")
    parser.add_argument("--input_col", required=True, help="Column name to use for calculation.")
    parser.add_argument("--output_col", required=True, help="Name of the new column to create.")
    
    args = parser.parse_args()

    # Step 1: Find the file path
    path_to_process = get_csv_path(args.manifest, args.line_number)
    
    if not path_to_process:
        logging.error(f"Could not find a path at line {args.line_number} in {args.manifest}")
        return
    
    if not os.path.exists(path_to_process):
        logging.error(f"File at {path_to_process} does not exist on disk.")
        return

    # Step 2: Get the year of the specific galaxy catalog configuration. 
    # This is needed for the redshift computation.
    label = path_to_process.split("prod_")[-1].split("/")[0]
    if label[1:3] == "10":
        year = 10
    elif label[1] == "7":
        year = 7
    elif label[1] == "4":
        year = 4
    else: 
        year = 1
        
    # Step 3: Run the processing
    add_column_to_csv(path_to_process, args.input_col, args.output_col, year)

if __name__ == "__main__":
    main()
