# 3. Analysis and Scoring

This directory contains the script for scoring the auto-interpretation. The main script, `scoring.py`, calculates the Normalized Discounted Cumulative Gain (NDCG) to evaluate the performance of Sparse Autoencoder (SAE) models. It compares the similarity of SAE features based on their co-occurrence patterns (Phi matrix) with the similarity of their embeddings from a separate model.

## Data Requirements

Before running the script, it is necessary to **download the required data** from the following Google Drive folder:

👉 [Download Data](https://drive.google.com/drive/folders/1uhg73Ecrmt0mCsXPYaZ0YejyHGbWW36Z?usp=drive_link)

After downloading, ensure that the **path specified in `config.yaml`** points to the correct location of the downloaded data on your system. Otherwise, the script will not be able to access the files.

## Configuration (`config.yaml`)

The `scoring.py` script uses `config.yaml` for configuration. This file allows you to set parameters for the scoring process without modifying the script itself.

- **`general`**: General settings like the random seed, the number of top features to consider (`top_n`), and the device to use (`cpu` or `cuda`).
- **`columns`**: Defines the names of columns used in the dataframes.
- **`paths`**: Specifies the base directory for the data (make sure it matches the location where you placed the downloaded data).
- **`models`**: Contains configurations for different models like `llama` and `gemma`. For each model, you can define the base directory, the layers to be analyzed, patterns for file names, and other model-specific parameters.

## Scoring Script: `scoring.py`

### How to Run
The script is run from the command line and requires specifying the model, layers, and embeddings to use.

### Arguments
- `--model`: The SAE model to use (e.g., `llama`, `gemma`). The available models are defined in `config.yaml`.
- `--layers`: A list of layer indices to process (e.g., `0 8 17`).
- `--embeddings`: The embedding models to use for comparison (e.g., `'Qwen/Qwen2-7B-Instruct'`).
- `--output`: The path to save the output CSV file (default: `scoring_results.csv`).

### Example
```bash
python 3_analysis/scoring.py --model llama --layers 0 8 17 --embeddings "Qwen/Qwen2-7B-Instruct" --output my_llama_scores.csv

```
