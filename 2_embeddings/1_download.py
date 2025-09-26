#!/usr/bin/env python
"""
Neuronpedia Data Collection
"""
# Import from pipeline package
from neuronpedia.neuronpedia_handler import download_explanations_from_s3


# Import config from the local package
from config import ( 
    NEURONPEDIA_OUTPUT_FILE,
    MODEL_ID,
    SAE_ID,

)

def main():
    """Main function to run the script."""

   
    download_explanations_from_s3(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        output_path=NEURONPEDIA_OUTPUT_FILE
    )
    
if __name__ == "__main__":
    main()
