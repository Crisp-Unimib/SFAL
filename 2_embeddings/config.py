"""
Configuration file for the Neuronpedia explanation downloader script.
"""
import os
import dotenv

# Load environment variables from a .env file if it exists.
dotenv.load_dotenv()

# --- API Configuration ---
# Your personal API key for Neuronpedia. Set this in your .env file.
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_KEY")

# --- Model & SAE Identifiers ---
# These are used to specify which explanations to download via the API.
MODEL_ID = "gemma-2-9b"
SAE_ID = "41-gemmascope-res-16k"

# --- Output File Configuration ---
# This is used to create an organized folder structure for the output data.
SAE_RELEASE_NAME = "gemmascope-res-16k"

# Create a clean directory path, e.g., 'data/Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L0R-8x/'
# The script will ensure this directory exists before saving the file.
OUTPUT_DIRECTORY = os.path.join('data', SAE_RELEASE_NAME, SAE_ID.replace('/', '_'))

# Define the full path for the output CSV file.
NEURONPEDIA_OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, 'explanations.csv')

# --- S3 Configuration ---
S3_BUCKET_NAME = "neuronpedia-datasets"

# Embeddding configuration
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-large"
EMBEDDING_BATCH_SIZE = 16
AUTOINTERP_MODELS = ['gpt-4o-mini',]
AUTOINTERP_TYPE = "oai_token-act-pair"
EMBEDDING_OUTPUT_PATH = f'data/{SAE_RELEASE_NAME}/{SAE_ID.replace('/', '.')}/{EMBEDDING_MODEL}/{AUTOINTERP_TYPE}_{AUTOINTERP_MODELS[0]}_embeddings.pkl'