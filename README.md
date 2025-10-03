# SFAL: Semantic–Functional Alignment Scores

This repository contains a pipeline for analyzing and linking features from Sparse Autoencoders (SAEs). The process involves calculating the co-occurrence of SAE features, generating embeddings for their explanations, and scoring the alignment between co-occurrence patterns and semantic similarity.

## Repository Structure

The repository is organized into three main stages:

- **`1_cooc_matrix/`**: Scripts to calculate the co-occurrence matrix of SAE features for a given model and corpus.
- **`2_embeddings/`**: A pipeline to download SAE feature explanations from Neuronpedia and generate embeddings for them.
- **`3_analysis/`**: Scripts to perform analysis and scoring of the auto-interpretation by comparing co-occurrence similarity with embedding similarity.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SFAL.git
    cd SFAL
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

The following sections describe how to run each step of the pipeline.

### Step 1: Co-occurrence Matrix Calculation

This step calculates the co-occurrence of SAE features on a given model and corpus. The main script is `1_cooc_matrix/run_cooc.py`.

**Usage:**

The script uses sub-commands for different model families (e.g., `gemma`, `llama`).

**Common Arguments:**

- `--model`: Hugging Face model ID (e.g., `google/gemma-2-2b`).
- `--layers`: List of layer indices to process.
- `--n_docs`: Number of documents from the corpus to use (default: 10,000).
- `--save_dir`: Directory to save the output `.npz` histograms (default: `histograms`).
- `--k`: Context window size for co-occurrence counting (default: 256).

**Example (Gemma):**

```bash
python 1_cooc_matrix/run_cooc.py gemma \
    --model google/gemma-2-2b \
    --layers 0 \
    --n_docs 100 \
    --save_dir ./cooc \
    --k 256 \
    --sae_features 16k \
    --target_l0 105 \
    --sae_type res
```

**Example (Llama):**

```bash
python 1_cooc_matrix/run_cooc.py llama \
    --model meta-llama/Llama-3.1-8B \
    --layers 0 \
    --n_docs 100 \
    --save_dir ./cooc \
    --k 256 \
    --top_k 50 \
    --sae_type res
```

### Step 2: Generating Embeddings

This step downloads SAE explanations from Neuronpedia and generates embeddings for them. The main script is `2_embeddings/run.py`.

**Configuration:**

Modify `2_embeddings/config.py` to set the `MODEL_ID`, `SAE_ID`, and `EMBEDDING_MODEL`.

**Usage:**

You can run the pipeline as a script from the `2_embeddings` directory or as a module from the root directory.

**Run as a script:**

```bash
cd 2_embeddings
# Download explanation data
python run.py download

# Generate embeddings
python run.py embed

# Run both steps
python run.py all
```

**Run as a module:**

```bash
python -m 2_embeddings download
python -m 2_embeddings embed
python -m 2_embeddings all
```

### Step 3: Analysis and Scoring

This step calculates the Normalized Discounted Cumulative Gain (NDCG) to evaluate the alignment between feature co-occurrence and semantic similarity of their explanations.

**Data Requirement:**

You must download the required data from this [Google Drive folder](https://drive.google.com/drive/folders/1uhg73Ecrmt0mCsXPYaZ0YejyHGbWW36Z?usp=drive_link) and place it in a local directory.

**Configuration:**

Update the `paths.base` key in `3_analysis/config.yaml` to point to the directory where you downloaded the data.

**Usage:**

The main script is `3_analysis/scoring.py`.

**Arguments:**

- `--model`: The SAE model to use (`llama` or `gemma`).
- `--layers`: List of layer indices to process.
- `--embeddings`: The embedding models to use for comparison.
- `--output`: Path to save the output CSV file (default: `scoring_results.csv`).

**Example:**

```bash
python 3_analysis/scoring.py \
    --model llama \
    --layers 0 8 17 \
    --embeddings "Qwen/Qwen2-7B-Instruct" \
    --output my_llama_scores.csv
```

## License

This project is licensed under the terms of the LICENSE file.
