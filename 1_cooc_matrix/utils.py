import os
import re
import numpy as np
import torch


def update_cooc_histogram(binary_acts, histogram, n_chunks, k):
    """
    Updates the co-occurrence histogram for a batch of activations.
    """
    for j in range(0, binary_acts.shape[0], k):
        if j + k <= binary_acts.shape[0]:
            chunk = binary_acts[j:j+k]
            chunk_features = torch.any(chunk, dim=0)
            co_occurrences = torch.outer(chunk_features, chunk_features)
            histogram += co_occurrences.int()
            n_chunks += 1
    return histogram, n_chunks

def closest_l0_name(sae_names, layer, width, l0):
    """
    Given a list of SAE names, like `layer_0/width_16k/average_l0_50/params.npz`, 
    returns the SAE name at that layer, with that width, with the closest l0 to the given l0.
    (Used for Gemma SAEs)
    """
    layer = str(layer)
    width = str(width)
    pattern = rf"^layer_{re.escape(layer)}/width_{re.escape(width)}/average_l0_(\d+)/params\.npz$"
    l0s = []
    for sae_name in sae_names:
        match = re.match(pattern, sae_name)
        if match:
            l0s.append(int(match.group(1)))
    if not l0s:
        raise ValueError(f"No SAEs found for layer {layer} and width {width}.")
    closest_l0 = l0s[np.argmin(np.abs(np.array(l0s) - l0))]
    return f"layer_{layer}/width_{width}/average_l0_{closest_l0}"

def select_sae_name(sae_names, layer_number, layer_token_suffix="R"):
    """
    Selects the correct SAE checkpoint filename from a list based on the layer number.
    (Used for Llama SAEs)
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


