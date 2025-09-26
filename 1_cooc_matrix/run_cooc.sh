#!/bin/bash

#SBATCH --account=IscrC_MI-PLE
#SBATCH --job-name=cooc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --array=8,17,31
#SBATCH --output=logs/cooc_%A_%a.out
#SBATCH --error=logs/cooc_%A_%a.err

# --- ENVIRONMENT SETUP ---
echo "INFO: Loading modules..."
module load python/3.11.6--gcc--8.5.0
echo "SUCCESS: Modules loaded successfully."

echo "INFO: Setting up environment variables..."
export HF_HOME=/leonardo_work/IscrC_MI-PLE/dpoterti/hf
echo "  -> HF_HOME set to: $HF_HOME"

export HF_HUB_OFFLINE=1
echo "  -> HF_HUB_OFFLINE set to: $HF_HUB_OFFLINE"

echo "INFO: Changing to work directory..."
cd "$WORK/dpoterti/scoring_autoint_align" || { echo "ERROR: Failed to change directory"; exit 1; }
echo "  -> Current directory: $(pwd)"

echo "INFO: Activating virtual environment..."
source env/bin/activate || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
echo "SUCCESS: Virtual environment activated."

echo "INFO: Goto cooc_matrix directory..."
cd 1_cooc_matrix || { echo "ERROR: Failed to change to cooc_matrix directory"; exit 1; }
echo "  -> Current directory: $(pwd)"

# --- JOB SETUP ---
echo "INFO: Creating logs directory if it doesn't exist..."
mkdir -p logs
echo "  -> Logs directory is present."

# #############################################################
# ##                                                         ##
# ##      LAYER IS SET BY THE SLURM ARRAY TASK ID            ##
# ##                                                         ##
# #############################################################
LAYER_TO_PROCESS=$SLURM_ARRAY_TASK_ID

# #############################################################
# ##                                                         ##
# ##      STAGGERED START TO AVOID FILE SYSTEM RACE          ##
# ##                                                         ##
# #############################################################
# This is the key change. We calculate a delay based on the task ID.
# We need to map the array values (8, 17, 31) to indices (0, 1, 2).
# A simple way is to use a case statement or an associative array.

declare -A layer_to_index
layer_to_index[8]=0
layer_to_index[17]=1
layer_to_index[31]=2

TASK_INDEX=${layer_to_index[$SLURM_ARRAY_TASK_ID]}
DELAY_SECONDS=$((TASK_INDEX * 180)) # 0*180=0s, 1*180=3min, 2*180=6min

echo "===================================================="
echo "          STARTING SLURM JOB ARRAY TASK"
echo "===================================================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "SLURM ARRAY JOB ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID (Layer: $LAYER_TO_PROCESS)"
echo "Task index: $TASK_INDEX"
echo "Calculated delay: $DELAY_SECONDS seconds"
echo "----------------------------------------------------"

echo "INFO: Sleeping for $DELAY_SECONDS seconds before starting..."
sleep $DELAY_SECONDS
echo "INFO: Sleep finished. Proceeding with execution."


# --- EXECUTION ---
python -u run_cooc.py llama \
    --model meta-llama/Llama-3.1-8B \
    --save_dir cooc_histograms/ \
    --n_docs 50000 \
    --layers "$LAYER_TO_PROCESS" \
    --k 256 \
    --sae_type res \
    --top_k 50


echo "===================================================="
echo "          SLURM JOB FINISHED"
echo "===================================================="