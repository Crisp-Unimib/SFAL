from pathlib import Path
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import reduce
import warnings

# --- Configuration ---
SEED = 28
TOP_N = 1023
BATCH_SIZE = 32 # Batch size for the reranker model
BASE_DATA_DIR = "/home/mmezzanzanica/project/scoring_autoint_align/data"
EMBEDDING_COL = "embedding"
LAYER_COL = "layer"
# OPTIMIZATION: Setting to control chunking for memory-intensive operations
PROCESSING_CHUNK_SIZE = 1024


# --- Seed everything for reproducibility ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

seed_everything(SEED)

# --- Device and Data Type Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {DEVICE} with dtype: {TORCH_DTYPE}")
else:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32
    print("CUDA not available. Using CPU. Performance will be significantly lower.")


MODELS = {
    "llama": {
        "base_dir": "Llama3_1-8B-Base-LXR-8x",
        "layers": [0, 8, 17, 25, 31],
        "layer_dir_pattern": "{layer}-llamascope-res-32k",
        "npz_filename": "pajama_meta-llama_Llama-3.1-8B_res_Llama3_1-8B-Base-L{layer}R-8x_checkpoints_final.safetensors_docs50k_keq256_cooccurrences.npz",
        "n_total_chunks": 71687,
        "feature_count": 32768,
    },
}

def load_filter_set(model_key: str):
    """Loads the set of (index, layer) combinations to filter calculations."""
    if model_key == "llama":
        filter_path = "/home/mmezzanzanica/project/scoring_autoint_align/3_analysis/eval/llama/score_llama.csv"
        df = pd.read_csv(filter_path)
        if "index" not in df or "layer" not in df:
            raise ValueError("LLaMA filter file must contain 'index' and 'layer' columns")
        return set(zip(df["index"].astype(int), df["layer"].astype(int)))
    else:
        raise ValueError(f"Model '{model_key}' not supported for filtering.")

def init_reranker(model_name="Qwen/Qwen3-Reranker-4B", device=DEVICE):
    """Initializes the reranker model and tokenizer in half-precision."""
    print(f"Initializing reranker model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=TORCH_DTYPE,
        device_map=device,
        attn_implementation="flash_attention_2"
    ).eval()

    # This compilation can be removed if it causes issues or is not supported in the environment.
    # It is intended for performance optimization.
    try:
        print("Compiling model with torch.compile()... (this may take a moment on the first run)")
        model = torch.compile(model)
    except Exception as e:
        print(f"Could not compile model: {e}. Proceeding without compilation.")


    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix = "<|im_start|>system\nYour task is to act as a semantics expert. Judge whether the Document is *conceptually and semantically related* to the Query, even if the words used are different. *Ignore superficial or structural similarity* and focus on the deep meaning. Your answer must be only 'yes' or 'no'.<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    print("Reranker initialized.")
    return tokenizer, model, token_true_id, token_false_id, prefix_tokens, suffix_tokens

@torch.no_grad()
def compute_rerank_score_batched(queries, docs, tokenizer, model, token_true_id, token_false_id, prefix_tokens, suffix_tokens, max_length=8192):
    """Processes a batch of query-document pairs for reranking."""
    prompts = [f"<Query>: {q}\n<Document>: {d}" for q, d in zip(queries, docs)]
    inputs = tokenizer(prompts, padding=False, truncation='longest_first',
                         return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
    for i in range(len(inputs['input_ids'])):
        inputs['input_ids'][i] = prefix_tokens + inputs['input_ids'][i] + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    logits = model(**inputs).logits[:, -1, :]
    true_scores = logits[:, token_true_id]
    false_scores = logits[:, token_false_id]
    stacked = torch.stack([false_scores, true_scores], dim=1)
    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
    return log_probs[:, 1].exp().cpu().tolist()

def calculate_top_phi_chunked(cooc_mat_np: np.ndarray, n_total: int, top_k: int, device=DEVICE, chunk_size=PROCESSING_CHUNK_SIZE):
    """Calculates top_k Phi correlations in chunks to prevent CUDA OOM errors."""
    n = cooc_mat_np.shape[0]
    d_gpu = torch.from_numpy(np.diag(cooc_mat_np).astype(np.float32)).to(device)
    n0_gpu = torch.tensor(float(n_total), device=device) - d_gpu
    top_phi_indices_cpu = np.zeros((n, top_k), dtype=np.int32)
    pbar = tqdm(range(0, n, chunk_size), desc="Calculating Top-K Phi (Chunked)")
    for i in pbar:
        end = min(i + chunk_size, n)
        cooc_chunk = torch.from_numpy(cooc_mat_np[i:end].astype(np.float32)).to(device)
        d_chunk = d_gpu[i:end].unsqueeze(1)
        ii_chunk = d_chunk.expand(-1, n)
        d_all = d_gpu.unsqueeze(0)
        jj_all = d_all.expand(end - i, -1)
        n11_chunk = cooc_chunk
        n10_chunk = ii_chunk - n11_chunk
        n01_chunk = jj_all - n11_chunk
        n00_chunk = torch.clamp(torch.tensor(float(n_total), device=device) - ii_chunk - jj_all + n11_chunk, min=0.0)
        n0_i_chunk = n0_gpu[i:end].unsqueeze(1)
        n0_j_all = n0_gpu.unsqueeze(0)
        prod = ii_chunk * jj_all * n0_i_chunk * n0_j_all
        denom = torch.sqrt(torch.clamp(prod, min=1e-12))
        phi_chunk = torch.zeros_like(n11_chunk)
        mask = prod >= 1e-12
        phi_chunk[mask] = ((n11_chunk[mask] * n00_chunk[mask]) - (n10_chunk[mask] * n01_chunk[mask])) / denom[mask]
        phi_chunk = torch.clamp(phi_chunk, -1.0, 1.0)
        torch.diagonal(phi_chunk, offset=0).fill_(-torch.inf) # Use offset=0 since we are working on a sub-matrix
        top_indices_chunk = torch.argsort(phi_chunk, dim=1, descending=True)[:, :top_k]
        top_phi_indices_cpu[i:end] = top_indices_chunk.cpu().numpy()
        del cooc_chunk, d_chunk, ii_chunk, jj_all, n11_chunk, n10_chunk, n01_chunk, n00_chunk, prod, denom, phi_chunk, mask, top_indices_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return top_phi_indices_cpu

def calculate_top_cosine_chunked(embs_np: np.ndarray, top_k: int, device=DEVICE, chunk_size=PROCESSING_CHUNK_SIZE):
    """Calculates top_k cosine similarities in chunks to prevent CUDA OOM errors."""
    n, _ = embs_np.shape
    embs_gpu = torch.from_numpy(embs_np.astype(np.float32)).to(device)
    embs_gpu = torch.nn.functional.normalize(embs_gpu, p=2, dim=1)
    top_cosine_indices_cpu = np.zeros((n, top_k), dtype=np.int32)
    pbar = tqdm(range(0, n, chunk_size), desc="Calculating Top-K Cosine Sim (Chunked)")
    for i in pbar:
        end = min(i + chunk_size, n)
        chunk_embs = embs_gpu[i:end]
        sim_chunk = torch.matmul(chunk_embs, embs_gpu.T)
        # To avoid selecting the same element, we set its similarity to -inf
        # Note: This is a simplification. For a perfect self-exclusion, a more complex indexing is needed.
        # But for large N, this is a reasonable approximation.
        torch.diagonal(sim_chunk, offset=i).fill_(-torch.inf)
        top_indices_chunk = torch.argsort(sim_chunk, dim=1, descending=True)[:, :top_k]
        top_cosine_indices_cpu[i:end] = top_indices_chunk.cpu().numpy()
        del chunk_embs, sim_chunk, top_indices_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del embs_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return top_cosine_indices_cpu

def find_embedding_pkls(layer_dir: Path):
    """Finds all embedding pickle files in the layer directory."""
    return {str(p.parent.relative_to(layer_dir)): p for p in layer_dir.rglob("oai_token-act-pair_gpt-4o-mini_embeddings.pkl")}

def process_layer(model_key, layer, model_conf, reranker_pack):
    print(f"\n=== {model_key.upper()} Layer {layer} ===")
    layer_dir = Path(BASE_DATA_DIR) / model_conf["base_dir"] / model_conf["layer_dir_pattern"].format(layer=layer)
    tokenizer, model, token_true_id, token_false_id, prefix_tokens, suffix_tokens = reranker_pack
    emb_models = find_embedding_pkls(layer_dir)
    if not emb_models:
        print(f"No embeddings found in {layer_dir}")
        return None

    npz_path = layer_dir / model_conf["npz_filename"].format(layer=layer)
    if not os.path.exists(npz_path):
        print(f"Cooc file not found for layer {layer}: {npz_path}")
        return None
    
    with np.load(npz_path) as npzf:
        cooc = npzf["histogram"]
    top_phi = calculate_top_phi_chunked(cooc, model_conf["n_total_chunks"], top_k=TOP_N, device=DEVICE)

    filter_set = load_filter_set(model_key)
    results_df = None

    for emb_name, pkl_path in emb_models.items():
        print(f"-> Processing embeddings: {emb_name}")
        df = pd.read_pickle(pkl_path).sort_values('index').reset_index(drop=True)
        
        # --- FIX START ---
        # Sanitize the 'description' column to ensure all entries are strings.
        # This prevents 'unhashable type: list' errors during the merge operation
        # by converting lists to strings and handling other non-string/null types.
        if 'description' in df.columns:
            df['description'] = df['description'].apply(
                lambda d: ' '.join(d) if isinstance(d, list) else str(d) if pd.notna(d) else ""
            )
        else:
            # If the column doesn't exist at all, create it with empty strings.
            df['description'] = ""
        # --- FIX END ---

        embedding_dim = len(df.loc[df['embedding'].notna(), 'embedding'].iloc[0])
        full_index_range = set(range(model_conf["feature_count"]))
        missing_indexes = sorted(full_index_range - set(df['index']))
        if missing_indexes:
            filler_rows = pd.DataFrame({
                'index': missing_indexes, 
                'embedding': [[0.0] * embedding_dim] * len(missing_indexes), 
                'description': [""] * len(missing_indexes)
            })
            df = pd.concat([df, filler_rows], ignore_index=True).sort_values('index').reset_index(drop=True)
        
        print(f"[INFO] Total embeddings loaded and filled: {len(df)}")
        filter_indices = {idx for idx, l in filter_set if l == layer}
        filtered_df = df[df["index"].isin(filter_indices)].reset_index(drop=True)
        
        if filtered_df.empty:
            print(f"[WARNING] No valid rows after filtering for layer {layer} ({model_key})")
            continue

        print(f"[INFO] Queries to process after filtering: {len(filtered_df)}")
        embs = np.stack(df[EMBEDDING_COL].values)
        top_cosine = calculate_top_cosine_chunked(embs, top_k=TOP_N, device=DEVICE)
        idx_map = {row["index"]: i for i, row in df.iterrows()}
        log2 = np.log2(np.arange(2, TOP_N + 2))
        ndcg_scores = {}

        pbar = tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"Reranking L{layer} [{emb_name}]")
        for _, row in pbar:
            idx = int(row["index"])
            i = idx_map[idx]
            query_text = row["description"] # Already a string due to sanitation
            candidate_indices = top_cosine[i]
            candidate_docs = [df.iloc[j]["description"] for j in candidate_indices]
            all_scores = []
            for k in range(0, len(candidate_indices), BATCH_SIZE):
                batch_queries = [query_text] * len(candidate_docs[k:k+BATCH_SIZE])
                batch_docs = candidate_docs[k:k+BATCH_SIZE]
                batch_scores = compute_rerank_score_batched(batch_queries, batch_docs, tokenizer, model, token_true_id, token_false_id, prefix_tokens, suffix_tokens)
                all_scores.extend(batch_scores)

            scored = list(zip(candidate_indices, all_scores))
            reranked_indices = [s[0] for s in sorted(scored, key=lambda x: -x[1])]
            sem_ranked_ids = df["index"].iloc[reranked_indices].to_numpy()
            phi_ranked_ids = df["index"].iloc[top_phi[i]].to_numpy()
            rel = {idj: 1.1 ** (TOP_N - rk) - 1 for rk, idj in enumerate(sem_ranked_ids)}
            dcg = sum(rel.get(j, 0) / log2[rk] for rk, j in enumerate(phi_ranked_ids))
            idcg_vals = [rel[j] for j in sem_ranked_ids if j in rel]
            idcg = sum(val / log2[rk] for rk, val in enumerate(sorted(idcg_vals, reverse=True)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores[idx] = ndcg
            pbar.set_postfix({"Last NDCG": f"{ndcg:.4f}"})

        df[f"ndcg_{emb_name}"] = df["index"].map(ndcg_scores)
        df[LAYER_COL] = layer
        keep_cols = ["index", "description", LAYER_COL, f"ndcg_{emb_name}"]
        current_results = df.loc[df[f"ndcg_{emb_name}"].notna(), keep_cols]

        if results_df is None:
            results_df = current_results
        else:
            # The merge should now work without type errors
            results_df = pd.merge(results_df, current_results, on=["index", "description", "layer"], how="outer")

    return results_df

def main():
    warnings.filterwarnings("ignore", message="`max_length` is ignored when `padding`=`True`")
    print("Running OPTIMIZED NDCG pipeline with reranking")
    reranker_pack = init_reranker()
    t0 = time.time()
    for model_key, model_conf in MODELS.items():
        print(f"\n### PROCESSING MODEL: {model_key.upper()} ###")
        all_layers_results = []
        for layer in model_conf["layers"]:
            layer_df = process_layer(model_key, layer, model_conf, reranker_pack)
            if layer_df is not None and not layer_df.empty:
                all_layers_results.append(layer_df)
        
        if all_layers_results:
            # FIX: The original code used reduce with pd.merge, which is incorrect for this task
            # and caused a MergeError. The correct approach is to stack the layer-specific
            # DataFrames vertically using pd.concat.
            final_df = pd.concat(all_layers_results, ignore_index=True)
            
            output_csv = f"optimized_reranked_ndcg_all_layers_{model_key}.csv"
            final_df.to_csv(output_csv, index=False)
            print(f"\n➡️  Saved final results to {output_csv}")
        else:
            print(f"No data produced for {model_key}")
            
    print(f"\n🏁 Completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
