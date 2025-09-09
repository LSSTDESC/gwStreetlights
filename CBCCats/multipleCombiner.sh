#!/bin/bash

# Directory containing the input files
INPUT_DIR="../data/"
# Name of the combined output file
OUTPUT_FILE="combined_batches.txt"

# Empty the output file if it already exists
> "$OUTPUT_FILE"

# Loop through files in numeric order from 0 to 255
for i in $(seq 0 255); do
    FILE="$INPUT_DIR/batch_0_bbh_aligned_${i}.txt"
    if [[ -f "$FILE" ]]; then
        cat "$FILE" >> "$OUTPUT_FILE"
    else
        echo "Warning: $FILE not found, skipping."
    fi
done

echo "All files combined into $OUTPUT_FILE"
