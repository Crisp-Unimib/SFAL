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
- `--sae_type`: The type of SAEs to process (`res`, `mlp`, `att`). Defaults to `res`.

**Example:**

```bash
python 1_cooc_matrix/run_cooc.py gemma --model google/gemma-2-2b --layers 0 --n_docs 100 --save_dir ./cooc --k 256 --sae_features 16k --target_l0 105 --sae_type res
```

#### Llama (`llama`)

To run with a Llama model, use the `llama` sub-command.

**Required Arguments:**

- `--sae_type`: The type of SAEs to process. Can be `res` (residual stream) or `mlp` (MLP output), or both (e.g., `res mlp`).
- `--top_k`: The 'k' value for the Top-K SAE activation.

**Example:**

```bash
python 1_cooc_matrix/run_cooc.py llama --model meta-llama/Llama-3.1-8B --layers 0 --n_docs 100 --save_dir ./cooc --k 256 --top_k 50 --sae_type res
```
