#!/usr/bin/env python
"""
Embedding Data Collection, 
"""
import os
# Import from pipeline package
from embeddings.embedding_handler import process_descriptions_and_compute_embeddings

# Import config from the local package
from config import ( 
    NEURONPEDIA_OUTPUT_FILE,
    EMBEDDING_OUTPUT_PATH,
    EMBEDDING_BATCH_SIZE
)

def main():
    """Main function to run the script."""
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
if __name__ == "__main__":
    main()
