#!/bin/bash

# --- Configuration ---

# Define the *single* layer you want to process
# <<< CHANGE THIS VALUE to the desired layer number >>>
LAYER_TO_PROCESS="0"

# Define base parameters for the python script
SAVE_DIR="/home/ec2-user/feature-geometry/brain/experiments/all-occurrences0/cooc-histograms/"
N_DOCS="50_000"
TARGET_L0="129"
K="256"
PYTHON_SCRIPT="/home/ec2-user/feature-geometry/brain/scripts/feature_cooccurrences_pile_all.py"
LOG_DIR="./logs" # Optional: Define a directory for logs

mkdir -p "$LOG_DIR" # Create log directory if it doesn't exist

# --- Execute the command for the single layer ---

echo "Starting job for layer: $LAYER_TO_PROCESS"

# Optional: Define a log file for this specific run
log_file="${LOG_DIR}/layer_${LAYER_TO_PROCESS}.log"

# Run the python script, passing the single layer number to --layers
# Redirect output and error to the log file (or remove > "$log_file" 2>&1 to see output directly)
python "$PYTHON_SCRIPT" \
    --save_dir "$SAVE_DIR" \
    --n_docs "$N_DOCS" \
    --layers "$LAYER_TO_PROCESS" \
    --target_l0 "$TARGET_L0" \
    --k "$K" > "$log_file" 2>&1 \


# Check exit status (optional but good practice)
if [ $? -eq 0 ]; then
    echo "Successfully finished job for layer: $LAYER_TO_PROCESS. Check log: $log_file"
else
    echo "ERROR occurred during job for layer: $LAYER_TO_PROCESS. Check log: $log_file"
    exit 1 # Exit with a non-zero status to indicate failure
fi

exit 0 # Explicitly exit with success status