#!/usr/bin/env python
"""
Main entry point for the embedding pipeline.
This script provides commands to download explanation data and compute embeddings.
"""
import argparse
import os

# Import from pipeline package
from .neuronpedia.neuronpedia_handler import download_explanations_from_s3
from .embeddings.embedding_handler import process_descriptions_and_compute_embeddings

# Import config from the local package
from .config import (
    NEURONPEDIA_OUTPUT_FILE,
    MODEL_ID,
    SAE_ID,
    EMBEDDING_OUTPUT_PATH,
    EMBEDDING_BATCH_SIZE
)

def download_data():
    """
    Downloads explanation data from Neuronpedia (S3).
    """
    print("Starting download of explanation data...")
    download_explanations_from_s3(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        output_path=NEURONPEDIA_OUTPUT_FILE
    )
    print("Download completed.")

def compute_embeddings():
    """
    Computes embeddings for the downloaded descriptions.
    """
    if not os.path.exists(NEURONPEDIA_OUTPUT_FILE):
        print(f"Explanation data not found at: {NEURONPEDIA_OUTPUT_FILE}")
        print("Please run the 'download' command first.")
        return

    if os.path.exists(EMBEDDING_OUTPUT_PATH):
        print(f"Embedding file already exists at: {EMBEDDING_OUTPUT_PATH}")
        print("Skipping description embedding computation.")
    else:
        print("Starting description processing and embedding computation...")
        process_descriptions_and_compute_embeddings(
            csv_path=NEURONPEDIA_OUTPUT_FILE,
            output_path=EMBEDDING_OUTPUT_PATH,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        print("Description embedding computation completed successfully.")

def main():
    """
    Main function to parse command-line arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Embedding Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Download command
    parser_download = subparsers.add_parser("download", help="Download explanation data from S3.")
    parser_download.set_defaults(func=download_data)

    # Embed command
    parser_embed = subparsers.add_parser("embed", help="Compute embeddings for the descriptions.")
    parser_embed.set_defaults(func=compute_embeddings)
    
    # All command
    parser_all = subparsers.add_parser("all", help="Run the full pipeline: download and embed.")
    
    def run_all(args):
        download_data()
        compute_embeddings()
    parser_all.set_defaults(func=run_all)


    args = parser.parse_args()
    if hasattr(args, 'func'):
        if args.command == 'all':
            args.func(args)
        else:
            args.func()

if __name__ == "__main__":
    main()
