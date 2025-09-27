import os
import json
from abc import ABC, abstractmethod

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# Imports from our new modules
from sae_models import JumpReLUSAE, TopKSAE
from utils import closest_l0_name, select_sae_name


class CoocHandler(ABC):
    """Abstract base class for a model/SAE handler."""

    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def load_model(self):
        """Loads the main model and tokenizer."""
        pass

    @abstractmethod
    def load_saes(self):
        """Loads all required SAEs."""
        pass
    
    @abstractmethod
    def get_target_act(self, cache, layer, sae_type):
        """Extracts the target activation from the model cache."""
        pass

    @abstractmethod
    def get_binary_acts(self, sae_acts, sae):
        """Converts SAE activations to binary firings."""
        pass

    @property
    @abstractmethod
    def dataset_id(self):
        """The Hugging Face dataset ID."""
        pass
    
    @property
    @abstractmethod
    def dataset_is_streaming(self):
        """Whether the dataset should be loaded in streaming mode."""
        pass


class GemmaHandler(CoocHandler):
    """Handler for Gemma models and JumpReLUSAEs."""

    def load_model(self):
        print("Loading Gemma model...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            self.args.model,
            tokenizer=tokenizer,
            dtype=self.dtype,
            fold_ln=True,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.to(self.device)
        return model

    def load_saes(self):
        print("Loading Gemma SAEs...")
        saes = {sae_type: [] for sae_type in self.args.sae_type}
        sae_names = {sae_type: [] for sae_type in self.args.sae_type}

        for sae_type in self.args.sae_type:
            repo_suffix = f"pt-{sae_type}"
            
            model_name = self.args.model.split("/")[-1]
            parts = model_name.split('-')
            if len(parts) == 3 and parts[0] == 'gemma': # gemma-2-2b case
                model_size = parts[1]
            elif len(parts) == 2 and parts[0] == 'gemma': # gemma-2b case
                model_size = parts[1]
            else:
                print(f"Warning: Could not reliably determine model size from {self.args.model}. Using fallback logic.")
                model_size = self.args.model.split('-')[-2]

            repo_id = f"google/gemma-scope-{model_size}-{repo_suffix}"
            print(f"[DEBUG] SAE Type: {sae_type}")
            print(f"[DEBUG] model_name: {model_name}")
            print(f"[DEBUG] parts: {parts}")
            print(f"[DEBUG] model_size: {model_size}")
            print(f"[DEBUG] repo_id: {repo_id}")
            
            try:
                available_saes = [f for f in list_repo_files(repo_id) if f.endswith('params.npz')]
            except Exception as e:
                print(f"Could not list files for {repo_id}. Skipping {sae_type} SAEs. Error: {e}")
                continue

            for layeri in self.args.layers:
                try:
                    # closest_l0_name expects paths ending in params.npz, and returns the dir name
                    sae_dir_name = closest_l0_name(available_saes, layeri, self.args.sae_features, self.args.target_l0)
                    
                    # The full SAE name includes the directory, which we'll use for bookkeeping
                    sae_full_name = f"{sae_dir_name}/params.npz"

                    path_to_params = hf_hub_download(repo_id=repo_id, filename=sae_full_name)
                    
                    params = np.load(path_to_params)
                    pt_params = {k: torch.from_numpy(v).to(self.device) for k, v in params.items()}
                    
                    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
                    sae.load_state_dict(pt_params)
                    sae.to(self.device, dtype=self.dtype)
                    sae.eval()
                    
                    saes[sae_type].append(sae)
                    # Store the directory name as the identifier, as before
                    sae_names[sae_type].append(sae_dir_name)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Could not load {sae_type} SAE for layer {layeri}: {e}")
                    continue
        
        return saes, sae_names

    def get_target_act(self, cache, layer, sae_type):
        if sae_type == "res":
            act = cache[f'blocks.{layer}.hook_resid_post']
        elif sae_type == "mlp":
            act = cache[f'blocks.{layer}.hook_mlp_out']
        elif sae_type == "att":
            act = cache[f'blocks.{layer}.attn.hook_z']
            _, seq_len, n_heads, d_head = act.shape
            act = act.reshape(1, seq_len, n_heads * d_head)
        else:
            raise ValueError(f"Unknown sae_type: {sae_type}")
        return act.to(self.device)

    def get_binary_acts(self, sae_acts, sae):
        # For JumpReLUSAE, the encoding step already applies a threshold,
        # so any non-zero activation is considered a "firing".
        return (sae_acts > 0).float()

    @property
    def dataset_id(self):
        return "monology/pile-uncopyrighted"

    @property
    def dataset_is_streaming(self):
        return True


class LlamaHandler(CoocHandler):
    """Handler for Llama models and TopKSAEs."""

    def load_model(self):
        print("Loading Llama model...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            self.args.model,
            tokenizer=tokenizer,
            dtype=self.dtype,
            fold_ln=True,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.to(self.device)
        return model

    def load_saes(self):
        print("Loading Llama SAEs...")
        sae_types_map = {
            "res": ("fnlp/Llama3_1-8B-Base-LXR-8x", "R"),
            "mlp": ("fnlp/Llama3_1-8B-Base-LXM-8x", "M"),
        }
        
        saes = {sae_type: [] for sae_type in self.args.sae_type}
        sae_names = {sae_type: [] for sae_type in self.args.sae_type}

        for sae_type in self.args.sae_type:
            if sae_type not in sae_types_map:
                continue
            
            repo_id, suffix = sae_types_map[sae_type]
            try:
                available_saes = [f for f in list_repo_files(repo_id) if f.endswith('final.safetensors')]
            except Exception as e:
                print(f"Could not list files for {repo_id}. Skipping {sae_type} SAEs. Error: {e}")
                continue

            for layeri in self.args.layers:
                try:
                    sae_name = select_sae_name(available_saes, layeri, layer_token_suffix=suffix)
                    sae_base_dir = os.path.dirname(os.path.dirname(sae_name))
                    
                    hyperparams_path = hf_hub_download(repo_id=repo_id, filename=os.path.join(sae_base_dir, 'hyperparams.json'))
                    path_to_params = hf_hub_download(repo_id=repo_id, filename=sae_name)

                    with open(hyperparams_path) as f:
                        hyperparams = json.load(f)
                    
                    k = self.args.top_k or hyperparams.get("k") or hyperparams.get("top_k")
                    if not k:
                        raise ValueError("Could not determine 'k' for TopKSAE.")

                    d_model = hyperparams["d_model"]
                    d_sae = hyperparams["d_sae"]

                    raw_tensors = load_file(path_to_params, device="cpu")
                    pt_params = {
                        'W_enc': raw_tensors['encoder.weight'].T,
                        'W_dec': raw_tensors['decoder.weight'].T,
                        'b_enc': raw_tensors.get('encoder.bias', torch.zeros(d_sae)),
                        'b_dec': raw_tensors.get('decoder.bias', torch.zeros(d_model)),
                    }

                    sae = TopKSAE(d_model, d_sae, k)
                    sae.load_state_dict(pt_params)
                    sae.to(self.device, dtype=self.dtype)
                    sae.eval()

                    saes[sae_type].append(sae)
                    sae_names[sae_type].append(sae_name)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Could not load {sae_type} SAE for layer {layeri}: {e}")
                    continue
        
        return saes, sae_names

    def get_target_act(self, cache, layer, sae_type):
        if sae_type == "res":
            act = cache[f'blocks.{layer}.hook_resid_post']
        elif sae_type == "mlp":
            act = cache[f'blocks.{layer}.hook_mlp_out']
        else:
            raise ValueError(f"Unknown sae_type: {sae_type}")
        return act.to(self.device)

    def get_binary_acts(self, sae_acts, sae):
        # For TopK, a feature "fired" if its activation is > 0.
        return (sae_acts > 0).float()

    @property
    def dataset_id(self):
        return "cerebras/SlimPajama-627B"

    @property
    def dataset_is_streaming(self):
        # The original script streams, but from a local path.
        # The load_dataset call needs to be adapted for this.
        return True