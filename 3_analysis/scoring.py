import os
import time
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

SEED = 28
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

DEVICE = torch.device("cpu")

BASE_DATA_DIR = "/home/mmezzanzanica/project/scoring_autoint_align/data"

MODELS = {
    "llama": {
        "base_dir": "Llama3_1-8B-Base-LXR-8x",
        "layers": [0, 8, 17, 25, 31],
        "layer_dir_pattern": "{layer}-llamascope-res-32k",
        "npz_filename": "pajama_meta-llama_Llama-3.1-8B_res_Llama3_1-8B-Base-L{layer}R-8x_checkpoints_final.safetensors_docs100k_keq512_cooccurrences.npz",
        "n_total_chunks": 44775,
    },
    # "gemma": {
    #     "base_dir": "gemmascope-res-16k",
    #     "layers": [0, 8, 17, 25, 41],
    #     "layer_dir_pattern": "{layer}-gemmascope-res-16k",
    #     # Qui serve una funzione lambda per il nome del file:
    #     "npz_filename": lambda layer: f"pile_google_gemma-2-9b_res_layer_{layer}_width_16k_average_l0_129_docs50k_keq256_cooccurrences.npz",
    #     "n_total_chunks": 70248,
    # }
}
TOP_N = 1023
EMBEDDING_COL = "embedding"
LAYER_COL = "layer"

def calculate_phi_matrix(cooc_mat_np: np.ndarray, n_total: int) -> torch.Tensor:
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

def find_embedding_pkls(layer_dir: Path) -> Dict[str, Path]:
    """
    Cerca ricorsivamente tutti i file oai_token-act-pair_gpt-4o-mini_embeddings.pkl nella directory di layer.
    Restituisce dict: {nome_embedding: path_pickle}
    """
    emb_pkls = {}
    for pkl_path in layer_dir.rglob("oai_token-act-pair_gpt-4o-mini_embeddings.pkl"):
        # emb_model = path relativa alla layer_dir, es: "Alibaba-NLP/gte-Qwen2-7B-instruct"
        rel_emb = pkl_path.parent.relative_to(layer_dir)
        emb_name = str(rel_emb)  # per chiarezza: Alibaba-NLP/gte-Qwen2-7B-instruct oppure Qwen/Qwen3-Embedding-8B
        emb_pkls[emb_name] = pkl_path
    return emb_pkls

def process_layer(model_key, layer, model_conf):
    print(f"\n=== {model_key.upper()} Layer {layer} ===")
    layer_dir = Path(BASE_DATA_DIR) / model_conf["base_dir"] / model_conf["layer_dir_pattern"].format(layer=layer)
    emb_models = find_embedding_pkls(layer_dir)
    if not emb_models:
        print(f"Nessun embedding trovato in {layer_dir}")
        return None

    # --- PATCH: logica di ricerca cooc per gemma ---
    if model_key == "gemma":
        npz_path = None
        for f in layer_dir.glob("*cooccurrences.npz"):
            if f"layer_{layer}_" in f.name:
                npz_path = f
                break
        if npz_path is None:
            print(f"Cooc file non trovato in {layer_dir} per layer {layer}")
            return None
    else:
        npz_name = model_conf["npz_filename"].format(layer=layer)
        npz_path = layer_dir / npz_name
        if not npz_path.exists():
            print(f"Cooc file non trovato: {npz_path}")
            return None

    # Carica cooccorrenze
    with np.load(npz_path) as npzf:
        cooc = npzf["histogram"]
    phi = calculate_phi_matrix(cooc, model_conf["n_total_chunks"])

    # Prepara struttura aggregata
    results_df = None
    # Per ogni modello di embedding
    for emb_name, pkl_path in emb_models.items():
        print('#'*20)
        print("per il modello:", model_key, "layer:", layer, "sto usando cooc file:", npz_path, "e embedding file:" , pkl_path)
        print('#'*20)
        print(f"-> Calcolo NDCG con embedding: {emb_name}")
        df = pd.read_pickle(pkl_path)
        df = df.sort_values('index').reset_index(drop=True)
        embedding_dim = len(df.loc[df['embedding'].notna(), 'embedding'].iloc[0])

        existing_indexes = set(df['index'])
        if model_key == "llama":
            full_index_range = set(range(0, 32768))
        elif model_key == "gemma":
            full_index_range = set(range(0, 16384))
        missing_indexes = sorted(full_index_range - existing_indexes)
        filler_rows = []
        for idx in missing_indexes:
            filler_rows.append({
                'id': np.nan, 'modelId': np.nan, 'layer': np.nan, 'index': idx, 'authorId': np.nan,
                'description': np.nan, 'embedding': [0.0] * embedding_dim, 'typeName': np.nan,
                'explanationModelName': np.nan, 'umap_x': np.nan, 'umap_y': np.nan,
                'umap_cluster': np.nan, 'umap_log_feature_sparsity': np.nan, 'true_description': np.nan
            })
        df_fillers = pd.DataFrame(filler_rows)
        df_filled = pd.concat([df, df_fillers], ignore_index=True)
        df = df_filled.sort_values('index').reset_index(drop=True)

        embs = np.stack(df[EMBEDDING_COL].values).astype(np.float32)
        sem = cosine_similarity(embs)
        np.fill_diagonal(sem, -np.inf)

        n = len(df)
        top_sem = np.argsort(-sem, axis=1)[:, :TOP_N]
        top_phi = torch.argsort(phi, dim=1, descending=True)[:,:TOP_N].cpu().numpy()
        idxs = df.index.to_numpy()
        log2 = np.log2(np.arange(2, TOP_N+2))
        scores = {}
        for i in tqdm(range(n), desc=f"NDCG L{layer} [{emb_name}]"):
            sem_i = top_sem[i][top_sem[i] < len(idxs)]
            phi_i = top_phi[i][top_phi[i] < len(idxs)]
            sem_nb = idxs[sem_i]
            phi_nb = idxs[phi_i]
            rel = {idj: 1.1**(TOP_N-rk)-1 for rk, idj in enumerate(sem_nb)}
            dcg  = sum(rel.get(j, 0) / log2[rk] for rk, j in enumerate(phi_nb))
            idcg = sum(rel[j]        / log2[rk] for rk, j in enumerate(sem_nb))
            scores[idxs[i]] = dcg / idcg if idcg else 0.0

        df[f"ndcg_{emb_name}"] = df.index.map(scores)
        df[LAYER_COL] = layer

        keep_cols = ["index", "description", LAYER_COL, f"ndcg_{emb_name}"] #rimosso 'id', se crea problemi cura in altro modo
        if results_df is None:
            results_df = df[keep_cols]
        else:
            results_df = results_df.merge(
                df[["index", f"ndcg_{emb_name}"]],
                on="index",
                how="left"
            )
    return results_df

def main():
    print("""

 _   _  ___________ _____   _    _ _____   _____ _____ 
| | | ||  ___| ___ \  ___| | |  | |  ___| |  __ \  _  |
| |_| || |__ | |_/ / |__   | |  | | |__   | |  \/ | | |
|  _  ||  __||    /|  __|  | |/\| |  __|  | | __| | | |
| | | || |___| |\ \| |___  \  /\  / |___  | |_\ \ \_/ /
\_| |_/\____/\_| \_\____/   \/  \/\____/   \____/\___/           

          """)
    t0 = time.time()
    for model_key, model_conf in MODELS.items():
        print(f"\n### PROCESSO MODELLO: {model_key.upper()} ###")
        all_layers_df = []
        for layer in model_conf["layers"]:
            layer_df = process_layer(model_key, layer, model_conf)
            if layer_df is not None:
                all_layers_df.append(layer_df)
        if all_layers_df:
            final_df = pd.concat(all_layers_df, ignore_index=True)
            output_csv = f"NO_rerank_ndcg_all_layers_{model_key}.csv"
            final_df.to_csv(output_csv, index=False)
            print(f"➡️  Salvato {output_csv}")
        else:
            print(f"Nessun dato prodotto per {model_key}")
    print(f"\n🏁 Completato in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
