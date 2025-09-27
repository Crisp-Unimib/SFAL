# Co-occurrence Matrix Calculation

This directory contains scripts to calculate the co-occurrence matrix of SAE features on a given model and corpus.

## Main Script: `run_cooc.py`

The primary script is `run_cooc.py`, which is designed to be modular and support different model families through sub-commands. It will download the model, dataset, and SAEs from the Hugging Face Hub.

### How to Run

The script uses sub-commands for each model family (`gemma` or `llama`).

#### Common Arguments

These arguments are available for all model families:
- `--model`: The Hugging Face model ID (e.g., `google/gemma-2-2b`).
- `--layers`: A list of layer indices to compute co-occurrences for.
- `--n_docs`: The number of documents from the corpus to process (default: 10,000).
- `--save_dir`: Directory to save the output `.npz` histograms (default: `histograms`).
- `--k`: The context window size (in tokens) for co-occurrence counting (default: 256).

#### Gemma (`gemma`)

To run with a Gemma model, use the `gemma` sub-command.

**Required Arguments:**
- `--sae_features`: The number of features in the SAE (e.g., `16k`, `32k`).
- `--target_l0`: The target L0 norm to use when selecting the best SAE.

**Example:**
```bash
python 1_cooc_matrix/run_cooc.py gemma \
    --model google/gemma-2-2b \
    --layers 0 1 2 \
    --sae_features 16k \
    --target_l0 50 \
    --n_docs 10000
```

#### Llama (`llama`)

To run with a Llama model, use the `llama` sub-command.

**Required Arguments:**
- `--sae_type`: The type of SAEs to process. Can be `res` (residual stream) or `mlp` (MLP output), or both (e.g., `res mlp`).
- `--top_k`: The 'k' value for the Top-K SAE activation.

**Example:**
```bash
python 1_cooc_matrix/run_cooc.py llama \
    --model meta-llama/Llama-3.1-8B \
    --layers 15 16 \
    --sae_type res mlp \
    --top_k 128 \
    --n_docs 50000
```

<<<<<<< HEAD
=======
---

## SLURM Script Example: `run_cooc.sh`

The file `run_cooc.sh` is an example of a batch script for the **SLURM Workload Manager**, which is commonly used on HPC clusters to schedule jobs.

### Key Features of the Script:

- **Environment Setup**: It loads the necessary modules and sets environment variables for offline execution (`HF_HUB_OFFLINE=1`).
- **Job Array**: It uses a SLURM job array (`--array=8,17,31`) to submit multiple jobs at once. Each job processes a different layer, which is passed to the python script via the `$SLURM_ARRAY_TASK_ID` variable.
- **Staggered Start**: It includes a delay mechanism to prevent multiple jobs from accessing the file system simultaneously at the very start, which can cause issues on some shared file systems.
- **Execution**: It calls the main `run_cooc.py` script with the appropriate parameters for running the `llama` configuration on a specific layer.

This script is tailored for a specific cluster environment and its directives (e.g., `#SBATCH --account=...`, `#SBATCH --partition=...`) would need to be adapted for your own HPC setup.
>>>>>>> 7f225160cf90e3f684dfd84eba137baa1d6deae5
