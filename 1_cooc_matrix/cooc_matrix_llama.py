"""
Counts the co-occurrences of Top-K SAE features across a corpus.
This script is modified for offline execution on an HPC cluster.
It assumes all necessary models and datasets have been pre-downloaded
to the local Hugging Face cache.
"""
import os
import argparse
import re
import time
import json
from glob import glob

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

import datasets
import datasets.config
import datasets.utils.file_utils
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoConfig


class TopKSAE(nn.Module):
    """
    Defines the Top-K Sparse Autoencoder architecture.
    For each input token, it activates the 'k' features with the highest
    pre-activation scores.
    """
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        """
        Encodes input activations into sparse feature activations.
        """
        # Calculate pre-activations
        pre_acts = input_acts @ self.W_enc + self.b_enc
        
        # Find the top k pre-activations along the feature dimension
        top_k_values, top_k_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # Apply ReLU to only the top-k values
        relu_top_k_values = torch.relu(top_k_values)
        
        # Create a sparse tensor for the final activations
        acts = torch.zeros_like(pre_acts)
        
        # Place the positive top-k values back into the tensor
        acts.scatter_(-1, top_k_indices, relu_top_k_values)
        
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

def select_sae_name(sae_names, layer_number, layer_token_suffix="R"):
    """
    Selects the correct SAE checkpoint filename from a list based on the layer number.
    No changes needed here.
    """
    if not isinstance(layer_token_suffix, str) or not layer_token_suffix:
        raise ValueError("layer_token_suffix must be a non-empty string (e.g., 'R' or 'M').")

    target_layer_token = f"L{layer_number}{layer_token_suffix}"
    
    matching_files = []
    for name in sae_names:
        if not name.endswith("/checkpoints/final.safetensors"):
            continue
        
        path_parts = name.split('/')
        if len(path_parts) < 3:
            continue
        
        model_and_layer_dir = path_parts[-3]
        regex_pattern_for_token = rf"(?:^|[_-]){re.escape(target_layer_token)}(?:[_-]|$)"
        
        if re.search(regex_pattern_for_token, model_and_layer_dir):
            matching_files.append(name)

    if not matching_files:
        raise ValueError(
            f"No SAE/checkpoint found for layer {layer_number} (using token '{target_layer_token}') "
            f"in the provided list. Checked {len(sae_names)} names."
        )
    
    if len(matching_files) > 1:
        raise ValueError(
            f"Multiple SAEs/checkpoints found for layer {layer_number} (token '{target_layer_token}'): "
            f"{matching_files}. Please ensure unique identifiers or refine selection criteria."
        )

    return matching_files[0]

def get_local_repo_path(repo_id, repo_type="model"):
    """
    Constructs the local cache path for a Hugging Face repo.
    This helps find the files without making a network call.
    """
    if repo_type == "dataset":
        folder_name = "datasets--" + repo_id.replace("/", "--")
    else:
        folder_name = "models--" + repo_id.replace("/", "--")
    
    cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    
    repo_path = os.path.join(cache_dir, "hub", folder_name)
    print(f"DEBUG: Constructed local repo path: {repo_path}")
    if not os.path.exists(repo_path):
            raise FileNotFoundError(
                f"Local cache for repo '{repo_id}' not found at '{repo_path}'. "
                "Please run the `huggingface-cli download` commands on a login node first."
            )
    return repo_path


@torch.no_grad()
def main(args):
    start_time = time.time()
    print("====================================================")
    print("      STARTING TOP-K SAE CO-OCCURRENCE COUNTING")
    print("====================================================")
    print(f"INFO: Script initiated with arguments: {args}")

    DATASET_ID = "cerebras/SlimPajama-627B"
    MODEL_ID = args.model

    # --- Model and Tokenizer Loading ---
    print("\n--- Loading Model and Tokenizer ---")
    print(f"INFO: Attempting to load tokenizer for '{MODEL_ID}' from local files.")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=True
    )
    print("SUCCESS: Tokenizer loaded.")

    print(f"INFO: Attempting to load model '{MODEL_ID}' from local files.")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        local_files_only=True,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print("SUCCESS: Model loaded.")
    print(f"  -> Model is on device: {device}")
    print(f"  -> Model is using dtype: {dtype}")

    print(f"\nINFO: Processing layers: {args.layers}")
    print(f"INFO: Selected SAE type: {args.sae_type}")

    # --- Initialization ---
    res_saes, res_saes_names = [], []
    mlp_saes, mlp_saes_names = [], []
    res_cooc_histograms, mlp_cooc_histograms = {}, {}
    res_n_chunks, mlp_n_chunks = {}, {}

    # --- Load Residual SAEs ---
    if args.sae_type in ['res', 'both']:
        print("\n--- Loading Residual Stream SAEs (res) ---")
        repo_id = "fnlp/Llama3_1-8B-Base-LXR-8x"
        print(f"INFO: Finding local path for SAE repo '{repo_id}'...")
        local_repo_path = get_local_repo_path(repo_id)
        
        snapshot_dir_pattern = os.path.join(local_repo_path, 'snapshots', '*')
        snapshot_dir = glob(snapshot_dir_pattern)[0]
        print(f"  -> Found snapshot directory: {snapshot_dir}")

        search_pattern = os.path.join(snapshot_dir, '**', 'final.safetensors')
        res_saes_available = [os.path.relpath(p, snapshot_dir) for p in glob(search_pattern, recursive=True)]
        print(f"INFO: Found {len(res_saes_available)} available residual SAE files.")

        for layeri in args.layers:
            print(f"\nINFO: Processing layer {layeri} for 'res' SAE...")
            sae_name = select_sae_name(res_saes_available, layeri)
            print(f"  -> Selected SAE file: {sae_name}")

            sae_base_dir = os.path.dirname(os.path.dirname(sae_name))
            hyperparams_path = os.path.join(snapshot_dir, sae_base_dir, 'hyperparams.json')
            path_to_params = os.path.join(snapshot_dir, sae_name)
            print(f"  -> Hyperparameters file: {hyperparams_path}")
            print(f"  -> Parameters file: {path_to_params}")

            with open(hyperparams_path) as f:
                hyperparams = json.load(f)
            
            # Determine 'k' for TopK SAE
            if args.top_k is not None:
                k = args.top_k
                print(f"  -> Using 'k' from command-line: {k}")
            elif "k" in hyperparams:
                k = hyperparams["k"]
            elif "top_k" in hyperparams:
                k = hyperparams["top_k"]
            else:
                raise ValueError("Could not determine 'k' for TopKSAE. Not in hyperparams and --top_k not set.")

            d_model = hyperparams["d_model"]
            d_sae = hyperparams["d_sae"]
            print(f"  -> Loaded hyperparameters: d_model={d_model}, d_sae={d_sae}, k={k}")
            
            raw_loaded_tensors = load_file(path_to_params, device="cpu")
            mapped_pt_params = {
                'W_enc': raw_loaded_tensors['encoder.weight'].T,
                'W_dec': raw_loaded_tensors['decoder.weight'].T,
            }
            if 'encoder.bias' in raw_loaded_tensors:
                mapped_pt_params['b_enc'] = raw_loaded_tensors['encoder.bias']
            if 'decoder.bias' in raw_loaded_tensors:
                mapped_pt_params['b_dec'] = raw_loaded_tensors['decoder.bias']

            sae = TopKSAE(d_model, d_sae, k)
            sae.load_state_dict(mapped_pt_params)
            sae.to(device, dtype=dtype)
            sae.eval()
            res_saes.append(sae)
            res_saes_names.append(sae_name)
            print(f"SUCCESS: Loaded and initialized TopKSAE for layer {layeri}.")
    
        res_cooc_histograms = {
            sae_name: torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
            for sae_name, sae in zip(res_saes_names, res_saes)
        }
        res_n_chunks = {sae_name: 0 for sae_name in res_saes_names}

    # --- Load MLP SAEs ---
    if args.sae_type in ['mlp', 'both']:
        print("\n--- Loading MLP SAEs (mlp) ---")
        repo_id = "fnlp/Llama3_1-8B-Base-LXM-8x"
        print(f"INFO: Finding local path for SAE repo '{repo_id}'...")
        local_repo_path = get_local_repo_path(repo_id)
        
        snapshot_dir = glob(os.path.join(local_repo_path, 'snapshots', '*'))[0]
        print(f"  -> Found snapshot directory: {snapshot_dir}")

        search_pattern = os.path.join(snapshot_dir, '**', 'final.safetensors')
        mlp_saes_available = [os.path.relpath(p, snapshot_dir) for p in glob(search_pattern, recursive=True)]
        print(f"INFO: Found {len(mlp_saes_available)} available MLP SAE files.")

        for layeri in args.layers:
            print(f"\nINFO: Processing layer {layeri} for 'mlp' SAE...")
            sae_name = select_sae_name(mlp_saes_available, layeri, layer_token_suffix="M")
            print(f"  -> Selected SAE file: {sae_name}")

            sae_base_dir = os.path.dirname(os.path.dirname(sae_name))
            hyperparams_path = os.path.join(snapshot_dir, sae_base_dir, 'hyperparams.json')
            path_to_params = os.path.join(snapshot_dir, sae_name)
            print(f"  -> Hyperparameters file: {hyperparams_path}")
            print(f"  -> Parameters file: {path_to_params}")
            
            with open(hyperparams_path) as f:
                hyperparams = json.load(f)
            
            # Determine 'k' for TopK SAE
            if args.top_k is not None:
                k = args.top_k
                print(f"  -> Using 'k' from command-line: {k}")
            elif "k" in hyperparams:
                k = hyperparams["k"]
            elif "top_k" in hyperparams:
                k = hyperparams["top_k"]
            else:
                raise ValueError("Could not determine 'k' for TopKSAE. Not in hyperparams and --top_k not set.")
                
            d_model = hyperparams["d_model"]
            d_sae = hyperparams["d_sae"]
            print(f"  -> Loaded hyperparameters: d_model={d_model}, d_sae={d_sae}, k={k}")

            raw_loaded_tensors = load_file(path_to_params, device="cpu")
            mapped_pt_params = {
                'W_enc': raw_loaded_tensors['encoder.weight'].T,
                'W_dec': raw_loaded_tensors['decoder.weight'].T,
            }
            if 'encoder.bias' in raw_loaded_tensors:
                mapped_pt_params['b_enc'] = raw_loaded_tensors['encoder.bias']
            if 'decoder.bias' in raw_loaded_tensors:
                mapped_pt_params['b_dec'] = raw_loaded_tensors['decoder.bias']

            sae = TopKSAE(d_model, d_sae, k)
            sae.load_state_dict(mapped_pt_params)
            sae.to(device, dtype=dtype)
            sae.eval()
            mlp_saes.append(sae)
            mlp_saes_names.append(sae_name)
            print(f"SUCCESS: Loaded and initialized TopKSAE for layer {layeri}.")

        mlp_cooc_histograms = {
            sae_name: torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
            for sae_name, sae in zip(mlp_saes_names, mlp_saes)
        }
        mlp_n_chunks = {sae_name: 0 for sae_name in mlp_saes_names}
        
    loading_time = time.time() - start_time
    print(f"\nSUCCESS: All models and SAEs loaded in {loading_time:.2f} seconds.")
    
    # --- Dataset Loading ---
    print("\n--- Loading Dataset ---")
    print(f"INFO: Setting offline mode for datasets.")
    datasets.config.HF_DATASETS_OFFLINE = True
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    print(f"INFO: Finding local path for dataset '{DATASET_ID}'...")
    local_dataset_path = get_local_repo_path(DATASET_ID, repo_type="dataset")

    snapshot_dir_pattern = os.path.join(local_dataset_path, 'snapshots', '*')
    snapshot_dirs = glob(snapshot_dir_pattern)
    if not snapshot_dirs:
        raise FileNotFoundError(f"Snapshot directory not found in {local_dataset_path}. Did the download complete?")
    snapshot_dir = snapshot_dirs[0]
    print(f"  -> Found local dataset snapshot: {snapshot_dir}")

    datasets.utils.file_utils.is_offline_mode = lambda: True

    dataset = datasets.load_dataset(
        snapshot_dir,
        streaming=True, 
        split="train", 
        trust_remote_code=True,
        download_mode="reuse_cache_if_exists", 
        verification_mode="no_checks",
    )
    print("SUCCESS: Dataset loaded in streaming mode.")

    # --- Main Processing Loop ---
    print(f"\n--- Starting Document Processing Loop for {args.n_docs} documents ---")
    loop_t0 = time.time()
    for i, doc in tqdm(enumerate(dataset), total=args.n_docs, desc="Processing Documents"):
        if i >= args.n_docs:
            print(f"INFO: Reached document limit of {args.n_docs}. Stopping.")
            break
        try:
            inputs = model.tokenizer.encode(
                doc['text'], 
                return_tensors="pt", 
                add_special_tokens=True,
                max_length=1024,
                truncation=True
            ).to(device)
        except Exception as e:
            print(f"WARNING: Skipping document {i} due to tokenization error: {e}")
            continue

        if inputs.shape[1] == 0:
            print(f"WARNING: Skipping document {i} because it is empty after tokenization.")
            continue

        _, cache = model.run_with_cache(inputs)

        # Process residual SAEs
        if args.sae_type in ['res', 'both']:
            for layeri, sae_name, sae in zip(args.layers, res_saes_names, res_saes):
                target_act = cache[f'blocks.{layeri}.hook_resid_post']
                sae_acts = sae.encode(target_act)
                
                # For TopK, a feature "fired" if its activation is > 0.
                binary_acts = (sae_acts > 0).float()
                binary_acts = binary_acts[0, 1:] # remove BOS token

                for j in range(0, binary_acts.shape[0], args.k):
                    if j + args.k <= binary_acts.shape[0]:
                        chunk = binary_acts[j:j+args.k]
                        chunk_features = torch.any(chunk, dim=0)
                        co_occurrences = torch.outer(chunk_features, chunk_features)
                        res_cooc_histograms[sae_name] += co_occurrences.int()
                        res_n_chunks[sae_name] += 1

        # Process MLP SAEs
        if args.sae_type in ['mlp', 'both']:
            for layeri, sae_name, sae in zip(args.layers, mlp_saes_names, mlp_saes):
                target_act = cache[f'blocks.{layeri}.hook_mlp_out']
                sae_acts = sae.encode(target_act)

                # For TopK, a feature "fired" if its activation is > 0.
                binary_acts = (sae_acts > 0).float()
                binary_acts = binary_acts[0, 1:] # remove BOS token

                for j in range(0, binary_acts.shape[0], args.k):
                    if j + args.k <= binary_acts.shape[0]:
                        chunk = binary_acts[j:j+args.k]
                        chunk_features = torch.any(chunk, dim=0)
                        co_occurrences = torch.outer(chunk_features, chunk_features)
                        mlp_cooc_histograms[sae_name] += co_occurrences.int()
                        mlp_n_chunks[sae_name] += 1
    
    loop_tot = time.time() - loop_t0
    print(f"\nSUCCESS: Finished processing loop in {loop_tot:.2f} seconds ({loop_tot / args.n_docs:.3f} s/doc).")

    # --- Saving Histograms ---
    print("\n--- Saving Co-occurrence Histograms ---")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"INFO: Ensured save directory exists: {args.save_dir}")

    if args.sae_type in ['res', 'both']:
        print("INFO: Saving residual histograms...")
        for sae_name, histogram in res_cooc_histograms.items():
            filename = f"pajama_{args.model.replace('/', '_')}_res_{sae_name.replace('/', '_')}_docs{args.n_docs // 1_000}k_keq{args.k}_cooccurrences.npz"
            save_path = os.path.join(args.save_dir, filename)
            print(f"  -> Saving to: {save_path}")
            np.savez_compressed(
                save_path,
                histogram=histogram.cpu().numpy(),
                n_chunks=res_n_chunks[sae_name],
            )
            print(f"  -> Total chunks processed for this SAE: {res_n_chunks[sae_name]}")

    if args.sae_type in ['mlp', 'both']:
        print("INFO: Saving MLP histograms...")
        for sae_name, histogram in mlp_cooc_histograms.items():
            filename = f"pajama_{args.model.replace('/', '_')}_mlp_{sae_name.replace('/', '_')}_docs{args.n_docs // 1_000}k_keq{args.k}_cooccurrences.npz"
            save_path = os.path.join(args.save_dir, filename)
            print(f"  -> Saving to: {save_path}")
            np.savez_compressed(
                save_path,
                histogram=histogram.cpu().numpy(),
                n_chunks=mlp_n_chunks[sae_name],
            )
            print(f"  -> Total chunks processed for this SAE: {mlp_n_chunks[sae_name]}")

    total_time = time.time() - start_time
    print("\n====================================================")
    print(f"      SCRIPT FINISHED in {total_time:.2f} seconds")
    print("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counts Top-K SAE feature co-occurrences in an offline HPC environment.")
    parser.add_argument("--model", type=str, help="The model to use for feature extraction.", default="meta-llama/Llama-3.1-8B", choices=["meta-llama/Llama-3.1-8B"])
    parser.add_argument('--layers', nargs='+', type=int, required=True, help='List of integers specifying layer indices.')
    parser.add_argument("--sae_type", type=str, help="The type of SAEs to process.", default="both", choices=["res", "mlp", "both"])
    parser.add_argument("--n_docs", type=int, help="The number of documents to analyze.", default=10_000)
    parser.add_argument("--save_dir", type=str, help="The directory to save the histogram to.", default="histograms")
    parser.add_argument("--k", type=int, help="Co-occurrence memory window (in tokens).", default=256)
    parser.add_argument("--top_k", type=int, default=None, help="The 'k' for TopK SAE activation. Overrides value in hyperparams file if set.")
    
    # Custom parsing to handle single integer from SLURM
    raw_args = parser.parse_args()
    if isinstance(raw_args.layers, list) and len(raw_args.layers) == 1:
        try:
            layers_list = [int(x) for x in str(raw_args.layers[0]).split()]
            raw_args.layers = layers_list
        except (ValueError, TypeError):
            pass

    main(raw_args)