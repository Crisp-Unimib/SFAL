import os
import argparse
import time
import numpy as np
import torch
import datasets
from tqdm.auto import tqdm

# Local imports
from handlers import GemmaHandler, LlamaHandler
from utils import update_cooc_histogram

# A dictionary to map model families to their handlers
HANDLER_MAP = {
    "gemma": GemmaHandler,
    "llama": LlamaHandler,
}

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Calculates SAE feature co-occurrence matrices for different model families.")
    subparsers = parser.add_subparsers(dest="model_family", required=True, help="The family of model to use.")

    # --- Parent parser for common arguments ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model", type=str, required=True, help="The specific model ID to use.")
    parent_parser.add_argument('--layers', nargs='+', type=int, required=True, help='List of integers specifying layer indices.')
    parent_parser.add_argument("--n_docs", type=int, default=10_000, help="The number of documents to analyze.")
    parent_parser.add_argument("--save_dir", type=str, default="histograms", help="The directory to save the histograms to.")
    parent_parser.add_argument("--k", type=int, default=256, help="Co-occurrence memory window (in tokens).")

    # --- Gemma Subparser ---
    parser_gemma = subparsers.add_parser("gemma", parents=[parent_parser], help="Run for Gemma models")
    parser_gemma.add_argument("--sae_features", type=str, required=True, help="Number of features in SAE (e.g., '16k').")
    parser_gemma.add_argument('--target_l0', type=int, required=True, help='The target l0 to use when selecting SAEs.')
    parser_gemma.add_argument("--sae_type", nargs='+', default=["res"], choices=["res", "mlp", "att"], help="The type of SAEs to process. Default: [\"res\"]")

    # --- Llama Subparser ---
    parser_llama = subparsers.add_parser("llama", parents=[parent_parser], help="Run for Llama models")
    parser_llama.add_argument("--sae_type", nargs='+', default=["res"], choices=["res", "mlp"], help="The type of SAEs to process. Default: [\"res\"]")
    parser_llama.add_argument("--top_k", type=int, required=True, help="The 'k' for TopK SAE activation.")

    args = parser.parse_args()


    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"Using device: {device}, dtype: {dtype}")

    if args.model_family not in HANDLER_MAP:
        raise ValueError(f"Unknown model family: {args.model_family}. Available: {list(HANDLER_MAP.keys())}")

    handler = HANDLER_MAP[args.model_family](args, device, dtype)

    # --- Load Model and SAEs ---
    model = handler.load_model()
    saes_by_type, sae_names_by_type = handler.load_saes()

    # --- Initialize Histograms ---
    cooc_histograms = {}
    n_chunks = {}
    for sae_type, saes in saes_by_type.items():
        for i, sae in enumerate(saes):
            sae_name = sae_names_by_type[sae_type][i]
            if sae_name not in cooc_histograms:
                cooc_histograms[sae_name] = torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
                n_chunks[sae_name] = 0

    print("Loaded SAEs:")
    for name in cooc_histograms.keys():
        print(f"\t{name}")

    # --- Load Dataset ---
    print("Loading dataset...")
    dataset_kwargs = {
        "streaming": handler.dataset_is_streaming,
        "split": "train",
    }
    dataset_id = handler.dataset_id
    
    dataset = datasets.load_dataset(dataset_id, **dataset_kwargs)

    # --- Main Processing Loop ---
    loop_t0 = time.time()
    pbar = tqdm(enumerate(dataset), total=args.n_docs, desc="Processing Documents")
    for i, doc in pbar:
        if i >= args.n_docs:
            break
        
        try:
            inputs = model.tokenizer.encode(
                doc['text'], return_tensors="pt", add_special_tokens=True, max_length=1024, truncation=True
            ).to(model.cfg.device)
        except Exception as e:
            print(f"Skipping doc {i} due to tokenizer error: {e}")
            continue
        
        if inputs.shape[1] <= 1: # Need at least one token besides BOS
            continue

        _, cache = model.run_with_cache(inputs)

        for sae_type, saes in saes_by_type.items():
            if not saes: continue
            
            # This logic assumes saes are ordered by layer, which they are in the handlers.
            for sae_idx, sae in enumerate(saes):
                layer = args.layers[sae_idx] 
                sae_name = sae_names_by_type[sae_type][sae_idx]

                target_act = handler.get_target_act(cache, layer, sae_type)
                sae_acts = sae.encode(target_act)
                binary_acts = handler.get_binary_acts(sae_acts, sae)
                binary_acts = binary_acts[0, 1:]  # remove BOS token

                histogram = cooc_histograms[sae_name]
                current_n_chunks = n_chunks[sae_name]

                _, new_n_chunks = update_cooc_histogram(binary_acts, histogram, current_n_chunks, args.k)
                n_chunks[sae_name] = new_n_chunks

    loop_tot = time.time() - loop_t0
    print(f"Total processing time: {loop_tot:.2f} s")

    # --- Save Histograms ---
    os.makedirs(args.save_dir, exist_ok=True)
    for sae_name, histogram in cooc_histograms.items():
        dataset_name = handler.dataset_id.split('/')[0].lower()
        
        filename_parts = [
            dataset_name,
            args.model.replace('/', '_'),
            sae_name.replace('/', '_'),
            f"docs{args.n_docs // 1_000}k",
            f"keq{args.k}",
            "cooccurrences.npz"
        ]
        filename = "_".join(filename_parts)
        save_path = os.path.join(args.save_dir, filename)
        
        np.savez_compressed(
            save_path,
            histogram=histogram.cpu().numpy(),
            n_chunks=n_chunks[sae_name],
        )
        print(f"Saved histogram to {save_path}")

if __name__ == "__main__":
    main()
