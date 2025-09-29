# 2. Embeddings Pipeline

This directory contains the code for downloading SAE explanations from Neuronpedia and generating embeddings for them.

## Setup

1.  **Install dependencies**:
    Make sure you have installed the necessary Python packages from the main `requirements.txt` file in the root of the repository.
    ```bash
    pip install -r ../requirements.txt
    ```

2.  **Configuration**:
    The main configuration for this pipeline is in `config.py`. You can modify the `MODEL_ID`, `SAE_ID`, and `EMBEDDING_MODEL` variables in this file to change the target model and embedding method.

## How to Run the Pipeline

This pipeline is managed by `run.py` and can be executed from the command line.

You can run the pipeline steps individually or all at once.

### Run as a script

Navigate to this directory (`2_embeddings`) and run the following commands:

1.  **Download Data**:
    This command downloads the explanation data from the S3 bucket specified in the configuration.
    ```bash
    python run.py download
    ```

2.  **Generate Embeddings**:
    This command processes the downloaded data and computes embeddings for the descriptions.
    ```bash
    python run.py embed
    ```

3.  **Run the Full Pipeline**:
    To run both the download and embedding steps in sequence:
    ```bash
    python run.py all
    ```

### Run as a module

You can also run this pipeline as a module from the parent directory (`C:\Users\user\Documents\GitHub\SFAL`):

```bash
python -m 2_embeddings download
python -m 2_embeddings embed
python -m 2_embeddings all
```

## Output

-   **Downloaded Data**: The raw explanation data is saved as a CSV file in the `data/` directory, following the path defined in `config.py` (e.g., `data/gemmascope-res-16k/41-gemmascope-res-16k/explanations.csv`).
-   **Embeddings**: The computed embeddings are saved as a pickle file (`.pkl`) in the same directory structure (e.g., `data/gemmascope-res-16k/41-gemmascope-res-16k/Lajavaness/bilingual-embedding-large/oai_token-act-pair_gpt-4o-mini_embeddings.pkl`).
