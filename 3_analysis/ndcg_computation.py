# -*- coding: utf-8 -*-
"""
1_ndcg_computation.py

Per ogni layer:
 1. Carica embeddings (.pkl) e cooccorrenze (.npz).
 2. Calcola similarità semantica (cosine) e funzionale (Phi).
 3. Calcola NDCG tra i due ranking per ogni feature.
 4. Campiona fino a SAMPLE_PER_LAYER feature stratificate per decile.
 5. Produce:
    - 1.1_ndcg_all_layers.csv  (tutte le feature con NDCG)
    - final_sample_across_layers.csv (campione per layer)
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

import os
import random
import numpy as np
import torch

SEED = 28 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # solo per torch >=1.8 e CUDA deterministico

# (per torch >=1.8+)
torch.use_deterministic_algorithms(True)


# ── 1. Configurazioni globali ───────────────────────────────────────────────
BASE_DATA_DIR = "/home/mmezzanzanica/project/scoring_autoint_align/data"
MODEL_NAME      = "Llama3_1-8B-Base-LXR-8x"
EMBED_MODEL     = "Alibaba-NLP/gte-Qwen2-7B-instruct"
COOC_MODEL      = "pajama_meta-llama_Llama-3.1-8B_res_Llama3_1-8B-Base-L{layer}R-8x"
#/Users/antonioserino/Documents/scoring_autoint_align/data/Llama3_1-8B-Base-LXR-8x/0-llamascope-res-32k/Alibaba-NLP/gte-Qwen2-7B-instruct/oai_token-act-pair_gpt-4o-mini_embeddings.pkl
#/home/mmezzanzanica/project/scoring_autoint_align/data
PKL_PATH_TEMPLATE = (
    f"{BASE_DATA_DIR}/{MODEL_NAME}/{{layer}}-llamascope-res-32k/"
    f"{EMBED_MODEL}/oai_token-act-pair_gpt-4o-mini_embeddings.pkl"
)

NPZ_PATH_TEMPLATE = (
    f"{BASE_DATA_DIR}/{MODEL_NAME}/{{layer}}-llamascope-res-32k/"
    f"pajama_meta-llama_Llama-3.1-8B_res_Llama3_1-8B-Base-L{{layer}}R-8x_checkpoints_final.safetensors_docs50k_keq256_cooccurrences.npz"
)

NPZ_ARRAY_KEY    = "histogram"
LAYERS           = [0, 8, 17, 25, 31]
N_TOTAL_CHUNKS   = 70248
TOP_N            = 1023
RANDOM_STATE     = 42
SAMPLE_PER_LAYER = 20    # 2 per decile
MAX_PER_DECILE   = 1

# Colonne
EMBEDDING_COL    = "embedding"
ID_COL           = "id"
DESCRIPTION_COL  = "description"
OUTPUT_NDCG_COL  = "ndcg_emb_phi"
OUTPUT_DECILE_COL= "phi_decile"
OUTPUT_URL_COL   = "url"
LAYER_COL        = "layer"

DEVICE = torch.device("cpu")  # o "cuda" se preferisci


# ── 2. Funzioni helper ──────────────────────────────────────────────────────
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

def build_neuronpedia_url(idx: int, layer: int) -> str:
    nid = f"llama3.1-8b/{layer}-llamascope-res-32k"
    return (
        f"https://neuronpedia.org/{nid}/{idx}"
        "?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    )

def sample_deciles(df: pd.DataFrame) -> pd.DataFrame:
    nd = df[OUTPUT_NDCG_COL].dropna()
    if nd.empty:
        return pd.DataFrame()
    # decili
    try:
        df[OUTPUT_DECILE_COL] = pd.qcut(nd, q=10, labels=False, duplicates="drop")
    except ValueError:
        df[OUTPUT_DECILE_COL] = -1
    df[OUTPUT_DECILE_COL] = df[OUTPUT_DECILE_COL].astype(int)
    # fino a MAX_PER_DECILE per decile
    strat = df.groupby(OUTPUT_DECILE_COL, group_keys=False) \
              .apply(lambda g: g.sample(n=min(MAX_PER_DECILE, len(g)), random_state=RANDOM_STATE))
    # riempi fino a SAMPLE_PER_LAYER
    rem = SAMPLE_PER_LAYER - len(strat)
    if rem > 0:
        pool = df.drop(index=strat.index)
        extra = pool.sample(n=min(rem, len(pool)), random_state=RANDOM_STATE)
        strat = pd.concat([strat, extra])
    return strat.reset_index(drop=True)


# ── 3. Processamento di un singolo layer ────────────────────────────────────
def process_layer(layer: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Layer {layer} ===")
    pkl = Path(PKL_PATH_TEMPLATE.format(layer=layer))
    npz = Path(NPZ_PATH_TEMPLATE.format(layer=layer))
    df = pd.read_pickle(pkl)
    df = df.sort_values('index').reset_index(drop=True)

    # Trova dimensione embedding (assumendo che ogni embedding sia una lista/array)
    embedding_dim = len(df.loc[df['embedding'].notna(), 'embedding'].iloc[0])

    # Crea un set di tutti gli index attuali
    existing_indexes = set(df['index'])
    if MODEL_NAME == "Llama3_1-8B-Base-LXR-8x":
        # Per Llama3, gli index vanno da 0 a 32767
        full_index_range = set(range(0, 32768))
    elif MODEL_NAME == "gemma-2-9B":
        # Costruisci la lista completa di index attesi
        full_index_range = set(range(0, 16384))

    # Trova gli index mancanti
    missing_indexes = sorted(full_index_range - existing_indexes)

    # Costruisci righe fittizie
    filler_rows = []
    for idx in missing_indexes:
        filler_rows.append({
            'id': np.nan,
            'modelId': np.nan,
            'layer': np.nan,
            'index': idx,
            'authorId': np.nan,
            'description': np.nan,
            'embedding': [0.0] * embedding_dim,
            'typeName': np.nan,
            'explanationModelName': np.nan,
            'umap_x': np.nan,
            'umap_y': np.nan,
            'umap_cluster': np.nan,
            'umap_log_feature_sparsity': np.nan,
            'true_description': np.nan
        })

    # Crea un DataFrame dalle righe fittizie
    df_fillers = pd.DataFrame(filler_rows)

    # Unisci e riordina
    df_filled = pd.concat([df, df_fillers], ignore_index=True)
    df = df_filled.sort_values('index').reset_index(drop=True)
    # if ID_COL in df.columns and df.index.name != ID_COL:
    #     if df[ID_COL].is_unique:
    #         df = df.set_index(ID_COL, drop=False)
    #     else:
    #         raise ValueError("Colonna ID non è univoca, impossibile impostarla come indice.")
    print(f"la lunghezza del dataset con gli embedding è pari a {len(df)}")

    with np.load(npz) as npzf:
        cooc = npzf[NPZ_ARRAY_KEY]
    print(f"la lunghezza delle cooccorrenze è pari a {cooc.shape[0]}")
    # semantica
    embs = np.stack(df[EMBEDDING_COL].values).astype(np.float32)
    sem = cosine_similarity(embs)
    np.fill_diagonal(sem, -np.inf)
    # funzionale
    phi = calculate_phi_matrix(cooc, N_TOTAL_CHUNKS)

    # NDCG su tutte
    n = len(df)
    top_sem = np.argsort(-sem, axis=1)[:, :TOP_N]
    top_phi = torch.argsort(phi, dim=1, descending=True)[:,:TOP_N].cpu().numpy()
    idxs = df.index.to_numpy()
    log2 = np.log2(np.arange(2, TOP_N+2))
    scores = {}
    for i in tqdm(range(n), desc=f"NDCG L{layer}"):
        # Filtro gli indici invalidi
        sem_i = top_sem[i][top_sem[i] < len(idxs)]
        phi_i = top_phi[i][top_phi[i] < len(idxs)]

        sem_nb = idxs[sem_i]
        phi_nb = idxs[phi_i]
        rel = {idj: 1.1**(TOP_N-rk)-1 for rk, idj in enumerate(sem_nb)}

        dcg  = sum(rel.get(j, 0) / log2[rk] for rk, j in enumerate(phi_nb))
        idcg = sum(rel[j]        / log2[rk] for rk, j in enumerate(sem_nb))
        scores[idxs[i]] = dcg / idcg if idcg else 0.0


    df[OUTPUT_NDCG_COL] = df.index.map(scores)
    df[LAYER_COL] = layer

    # campione decili
    samp = sample_deciles(df)
    if not samp.empty:
        samp[OUTPUT_URL_COL] = samp["index"].apply(lambda i: build_neuronpedia_url(i, layer))
        samp[LAYER_COL] = layer

    return df, samp


# ── 4. Main ────────────────────────────────────────────────────────────────
def main() -> None:
    all_dfs: List[pd.DataFrame] = []
    all_samps: List[pd.DataFrame] = []
    t0 = time.time()

    for lyr in LAYERS:
        try:
            df_all, df_samp = process_layer(lyr)
            all_dfs.append(df_all)
            if not df_samp.empty:
                all_samps.append(df_samp)
        except Exception as e:
            print(f"Layer {lyr} error: {e}")

    # salvo tutti i ndcg
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("1.1_ndcg_all_layers.csv", index=False)
    print("➡️  Salvato 1.1_ndcg_all_layers.csv")

    # salvo il campione finale: commenta se non vuoi il campione
    # final = pd.concat(all_samps, ignore_index=True)
    # final.to_csv("final_sample_across_layers.csv", index=False)
    # print("➡️  Salvato final_sample_across_layers.csv")

    print(f"\n🏁 Completato in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
