

import os
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def load_config(path: str = "config.yaml") -> Dict:
    """
    Loads the YAML configuration file.

    Args:
        path: The path to the configuration file.

    Returns:
        A dictionary with the configuration.
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{path}' not found.")
        print("Ensure that the config.yaml file is in the same folder as the script.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error reading the YAML file: {e}")
        exit(1)

# Global setup
CONFIG = load_config()

DEVICE = CONFIG['general']['device'].lower()

if DEVICE == "cuda":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        print("CUDA not available, falling back to CPU.")
        DEVICE = "cpu"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

def setup_environment():
    """Sets the seeds for reproducibility from the config file."""
    seed = CONFIG['general']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

# --- CALCULATION FUNCTIONS ---
def calculate_phi_matrix(cooc_mat_np: np.ndarray, n_total: int) -> torch.Tensor:
    """Calculates the Phi matrix (Matthews correlation coefficient)."""
    n = cooc_mat_np.shape[0]
    cooc = torch.from_numpy(cooc_mat_np.astype(np.float32)).to(DEVICE)
    N = torch.tensor(float(n_total), device=DEVICE)

    d = torch.diag(cooc).clone()
    ii = d.unsqueeze(1).expand(-1, n)
    jj = d.unsqueeze(0)

    n11 = cooc
    n10 = ii - n11
    n01 = jj - n11
    n00 = torch.clamp(N - ii - jj + n11, min=0.0)

    n0 = N - d
    n0_i = n0.unsqueeze(1)
    n0_j = n0.unsqueeze(0)

    prod = ii * jj * n0_i * n0_j
    denom = torch.sqrt(torch.clamp(prod, min=1e-12))
    phi = ((n11 * n00) - (n10 * n01)) / denom
    phi[prod < 1e-12] = 0.0
    phi = torch.clamp(phi, -1.0, 1.0)
    torch.diagonal(phi).fill_(-torch.inf)
    return phi

# --- FILE MANAGEMENT FUNCTIONS ---
def find_embedding_pkls(layer_dir: Path) -> Dict[str, Path]:
    """Recursively finds all embedding .pkl files in a directory."""
    emb_pkls = {}
    for pkl_path in layer_dir.rglob("oai_token-act-pair_gpt-4o-mini_embeddings.pkl"):
        rel_emb = pkl_path.parent.relative_to(layer_dir)
        emb_name = str(rel_emb)
        emb_pkls[emb_name] = pkl_path
    return emb_pkls

def find_cooc_file(model_key: str, layer: int, layer_dir: Path, model_conf: Dict) -> Optional[Path]:
    """Finds the correct co-occurrence (.npz) file for a given model and layer."""
    if model_key == "gemma":
        for f in layer_dir.glob("*cooccurrences.npz"):
            if f"layer_{layer}_" in f.name:
                return f
        return None
    else:
        npz_name = model_conf["npz_filename_pattern"].format(layer=layer)
        npz_path = layer_dir / npz_name
        return npz_path if npz_path.exists() else None


def process_layer(model_key: str, layer: int, model_conf: Dict, selected_embeddings: List[str]) -> Optional[pd.DataFrame]:
    """
    Processes a single model layer to calculate NDCG scores.

    This function finds the required embedding and co-occurrence files, calculates
    the Phi matrix, computes NDCG scores, and returns the results as a DataFrame.

    Args:
        model_key: The key for the model being processed (e.g., 'llama').
        layer: The layer number to process.
        model_conf: The configuration dictionary for the model.
        selected_embeddings: A list of embedding names to process.

    Returns:
        A DataFrame with the NDCG scores for the layer, or None if processing fails.
    """
    print(f"\n=== Processing: {model_key.upper()} Layer {layer} ===")
    base_data_dir = CONFIG['paths']['base_data_dir']
    layer_dir = Path(base_data_dir) / model_conf["base_dir"] / model_conf["layer_dir_pattern"].format(layer=layer)

    available_embs = find_embedding_pkls(layer_dir)
    if not available_embs:
        print(f"Warning: No embedding .pkl files found in {layer_dir}. Skipping layer.")
        return None

    embeddings_to_process = {name: path for name, path in available_embs.items() if name in selected_embeddings}
    if not embeddings_to_process:
        print(f"Warning: None of the selected embeddings {selected_embeddings} were found for layer {layer}. Skipping layer.")
        return None

    npz_path = find_cooc_file(model_key, layer, layer_dir, model_conf)
    if not npz_path:
        print(f"Warning: Co-occurrence file not found for layer {layer}. Skipping layer.")
        return None

    with np.load(npz_path) as npzf:
        cooc = npzf["histogram"]
    phi = calculate_phi_matrix(cooc, model_conf["n_total_chunks"])

    results_df = None
    embedding_col = CONFIG['columns']['embedding']
    layer_col = CONFIG['columns']['layer']
    top_n = CONFIG['general']['top_n']

    for emb_name, pkl_path in embeddings_to_process.items():
        print(f"-> Calculating NDCG for embedding: {emb_name}")
        df = pd.read_pickle(pkl_path)
        df = df.sort_values('index').reset_index(drop=True)

        embedding_dim = len(df.loc[df[embedding_col].notna(), embedding_col].iloc[0])
        full_index_range = set(range(model_conf["vocab_size"]))
        missing_indexes = sorted(full_index_range - set(df['index']))

        if missing_indexes:
            filler_rows = [{'index': idx, embedding_col: [0.0] * embedding_dim} for idx in missing_indexes]
            df_fillers = pd.DataFrame(filler_rows)
            df = pd.concat([df, df_fillers], ignore_index=True).sort_values('index').reset_index(drop=True)

        embs = np.stack(df[embedding_col].values).astype(np.float32)
        sem = cosine_similarity(embs)
        np.fill_diagonal(sem, -np.inf)

        n = len(df)
        top_sem = np.argsort(-sem, axis=1)[:, :top_n]
        top_phi = torch.argsort(phi, dim=1, descending=True)[:, :top_n].cpu().numpy()

        log2 = np.log2(np.arange(2, top_n + 2))
        scores = {}
        for i in tqdm(range(n), desc=f"NDCG L{layer} [{emb_name}]", leave=False):
            sem_nb = top_sem[i]
            phi_nb = top_phi[i]
            rel = {idj: 1.1**(top_n - rk) - 1 for rk, idj in enumerate(sem_nb)}
            dcg = sum(rel.get(j, 0) / log2[rk] for rk, j in enumerate(phi_nb))
            idcg = sum(rel.get(j, 0) / log2[rk] for rk, j in enumerate(sem_nb))
            scores[i] = dcg / idcg if idcg else 0.0

        df[f"ndcg_{emb_name}"] = df.index.map(scores)
        df[layer_col] = layer

        keep_cols = ["index", "description", layer_col, f"ndcg_{emb_name}"]
        if results_df is None:
            results_df = df[keep_cols]
        else:
            results_df = results_df.merge(df[["index", f"ndcg_{emb_name}"]], on="index", how="left")

    return results_df

def get_cli_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="NDCG Scoring Script for SAE Models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=CONFIG['models'].keys(),
        help="The SAE model to use."
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        required=True,
        help="Layers to be processed (e.g., --layers 0 8)."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        nargs='+',
        required=True,
        help="Embedding models to be used (e.g., --embeddings 'Qwen/Qwen2-7B-Instruct')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scoring_results.csv",
        help="Output CSV file path (default: scoring_results.csv)."
    )
    return parser.parse_args()

# --- MAIN ---
def main():
    """Main function to run the scoring process."""
    print("""

 _____ _____ ___________ _____ _   _ _____  
/  ___/  __ \  _  | ___ \_   _| \ | |  __ \ 
\ `--.| /  \/ | | | |_/ / | | |  \| | |  \/ 
 `--. \ |   | | | |    /  | | | . ` | | __  
/\__/ / \__/\ \_/ / |\ \ _| |_| |\  | |_\ \ 
\____/ \____/\___/\_| \_|\___/\_| \_/\____/ 
                                            
                                            

""")
    t0 = time.time()
    setup_environment()
    args = get_cli_args()

    model_key = args.model
    model_conf = CONFIG['models'][model_key]

    # Validate selected layers
    for layer in args.layers:
        if layer not in model_conf["layers"]:
            print(f"\nError: Layer {layer} is not valid for model {model_key}.")
            print(f"Available Layers: {model_conf['layers']}")
            return

    print(f"\n### Selected Configuration ###")
    print(f"Model: {model_key.upper()}")
    print(f"Layers: {args.layers}")
    print(f"Embeddings: {args.embeddings}")
    print("#################################\n")

    all_layers_df = []
    for layer in sorted(args.layers):
        layer_df = process_layer(model_key, layer, model_conf, args.embeddings)
        if layer_df is not None:
            all_layers_df.append(layer_df)

    if all_layers_df:
        final_df = pd.concat(all_layers_df, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print(f"\n➡️  Results saved to: {args.output}")
    else:
        print("\nNo results were generated. Please check the configuration and file paths.")

    print(f"\n🏁 Execution completed in {time.time()-t0:.1f} seconds.")


if __name__ == "__main__":
    main()