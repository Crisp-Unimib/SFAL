"""
Handler module for computing embeddings from descriptions.
This module uses a strategy pattern to support different embedding models and methodologies.
"""

import ast
import os
from abc import ABC, abstractmethod

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import AUTOINTERP_MODELS, AUTOINTERP_TYPE, EMBEDDING_MODEL

# ==============================================================================
# 1. STRATEGY PATTERN: DEFINE EMBEDDER INTERFACE AND IMPLEMENTATIONS
# ==============================================================================

class BaseEmbedder(ABC):
    """Abstract Base Class for all embedding strategies."""
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.init_kwargs = kwargs

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer into memory."""
        pass

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int, **kwargs) -> list[list[float]]:
        """Compute embeddings for a list of texts."""
        pass


class TransformersEmbedder(BaseEmbedder):
    """
    Embedder for standard Hugging Face Transformer models (e.g., gte-Qwen2).
    Uses AutoModel and extracts the CLS token embedding.
    """
    def load_model(self):
        print(f"Loading generic Transformer model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")

        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
            )
            print("Model loaded successfully with device_map='auto'")
        except Exception as e:
            print(f"Error loading model at full precision: {e}. Attempting 8-bit quantization.")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                load_in_8bit=True,
            )
            print("Model loaded with 8-bit quantization.")
        
        if hasattr(self.model, "hf_device_map"):
            print("Model distribution:", self.model.hf_device_map)
        
        self.model.eval()

    def embed(self, texts: list[str], batch_size: int, **kwargs) -> list[list[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings (Transformers)"):
            batch_texts = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=8192
            )
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                batch_embeddings = outputs[0][:, 0].cpu()
            
            embeddings.extend(batch_embeddings.numpy().tolist())
            torch.cuda.empty_cache()
        return embeddings


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Embedder for models optimized for the sentence-transformers library (e.g., Qwen3-Embedding).
    """
    def load_model(self):
        print(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, **self.init_kwargs)
        print("SentenceTransformer model loaded successfully.")

    def embed(self, texts: list[str], batch_size: int, **kwargs) -> list[list[float]]:
        print(f"Embedding with SentenceTransformer (batch size: {batch_size})")
        
        prompt_name = kwargs.get("prompt_name")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            prompt_name=prompt_name,
            convert_to_numpy=True
        )
        return embeddings.tolist()


def get_embedder(model_name: str) -> BaseEmbedder:
    """
    Factory function to select the appropriate embedder based on the model name.
    """
    # Define the recommended settings for Qwen3 SentenceTransformer models once
    # to avoid repetition and ensure consistency.
    qwen3_st_config = {
        "model_kwargs": {
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "torch_dtype": torch.bfloat16
        },
        "tokenizer_kwargs": {"padding_side": "left"},
    }

    EMBEDDER_REGISTRY = {
        # Original model using the generic Transformers class
        "Alibaba-NLP/gte-Qwen2-7B-instruct": (TransformersEmbedder, {}),

        # Models using the SentenceTransformers class with the same optimal config
        "Qwen/Qwen3-Embedding-0.6B": (SentenceTransformersEmbedder, qwen3_st_config),
        "Qwen/Qwen3-Embedding-4B":   (SentenceTransformersEmbedder, qwen3_st_config),
        "Qwen/Qwen3-Embedding-8B":   (SentenceTransformersEmbedder, qwen3_st_config),

        "Lajavaness/bilingual-embedding-large": (SentenceTransformersEmbedder, {"trust_remote_code": True}),
    }

    if model_name in EMBEDDER_REGISTRY:
        EmbedderClass, kwargs = EMBEDDER_REGISTRY[model_name]
        print(f"Found '{model_name}' in registry. Using '{EmbedderClass.__name__}'.")
        return EmbedderClass(model_name, **kwargs)
    else:
        print(f"Warning: Model '{model_name}' not in registry. Falling back to generic 'TransformersEmbedder'.")
        return TransformersEmbedder(model_name)

# ==============================================================================
# 2. DATA LOADING AND PREPROCESSING (Unchanged)
# ==============================================================================

def get_prioritized_explanation(models, descriptions, types):
    """Selects the description corresponding to the highest-priority model available."""
    autointerp_models = [AUTOINTERP_MODELS] if isinstance(AUTOINTERP_MODELS, str) else AUTOINTERP_MODELS
    for priority_model in autointerp_models:
        if priority_model in models:
            idx = models.index(priority_model)
            if idx < len(types) and types[idx] == AUTOINTERP_TYPE:
                description = descriptions[idx] if idx < len(descriptions) else None
                if description:
                    return description, priority_model
    return None, None

def safe_literal_eval(val):
    """Safely evaluate a string as a Python literal."""
    try:
        evaluated = ast.literal_eval(val)
        return evaluated if isinstance(evaluated, list) else [evaluated]
    except (ValueError, SyntaxError):
        return [val]

def load_and_preprocess_descriptions(csv_path):
    """Load, filter, and preprocess descriptions from a CSV file."""
    df = pd.read_csv(csv_path)
    df['explanationModelName'] = df['explanationModelName'].apply(safe_literal_eval)
    df['description'] = df['description'].apply(safe_literal_eval)
    df['typeName'] = df['typeName'].apply(safe_literal_eval)
    print(f"Loaded {len(df)} total explanation rows from '{csv_path}'.")

    selected_explanations = df.apply(
        lambda row: get_prioritized_explanation(row['explanationModelName'], row['description'], row['typeName']),
        axis=1,
        result_type='expand'
    )
    df[['true_description', 'selected_model']] = selected_explanations
    
    initial_count = len(df)
    df.dropna(subset=['true_description'], inplace=True)
    print(f"Filtered out {initial_count - len(df)} rows that did not have a valid explanation model from AUTOINTERP_MODELS.")
    
    if df.empty:
        print("[WARNING] No valid explanations found after filtering. Returning empty results.")
        return pd.DataFrame(), []

    autointerp_models_list = [AUTOINTERP_MODELS] if isinstance(AUTOINTERP_MODELS, str) else AUTOINTERP_MODELS
    model_priority_map = {model: i for i, model in enumerate(autointerp_models_list)}
    df['model_priority'] = df['selected_model'].map(model_priority_map)
    
    df.sort_values(by=['index', 'model_priority'], ascending=[True, True], inplace=True)
    
    initial_unique_count = len(df)
    df.drop_duplicates(subset=['index'], keep='first', inplace=True)
    print(f"Removed {initial_unique_count - len(df)} duplicate explanations, keeping the highest-priority version.")
    
    df.drop(columns=['selected_model', 'model_priority'], inplace=True)
    
    documents_to_embed = df["true_description"].astype(str).tolist()
    
    print(f"Final dataset contains {len(df)} unique explanations to be embedded.")
    
    return df, documents_to_embed


# ==============================================================================
# 3. MAIN WORKFLOW (Unchanged)
# ==============================================================================

def process_descriptions_and_compute_embeddings(csv_path, output_path, batch_size=16):
    """
    Main function to process descriptions and compute embeddings using the selected model.
    """
    df, documents_to_embed = load_and_preprocess_descriptions(csv_path)
    
    if df.empty:
        print("Processing stopped as there are no descriptions to embed.")
        return False
        
    print(f"Loaded {len(documents_to_embed)} descriptions for embedding using model: {EMBEDDING_MODEL}")
    
    embedder = get_embedder(EMBEDDING_MODEL)
    
    embedder.load_model()
    
    embeddings = embedder.embed(documents_to_embed, batch_size=batch_size)
    df["embedding"] = embeddings
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    df.to_pickle(output_path)
    print(f"Embeddings computed and saved to '{output_path}'")
    
    return True