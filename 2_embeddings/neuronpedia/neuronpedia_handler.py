import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import logging
from tqdm import tqdm
import os
import gzip
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("s3_downloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants 
# import constants from the config file
from ..config import MODEL_ID, SAE_ID, S3_BUCKET_NAME

def download_explanations_from_s3(model_id: str, sae_id: str, output_path: str):
    """
    Downloads all explanation files for a given model and SAE from the public
    S3 bucket, merges them, and saves them to a single CSV file.

    Args:
        model_id (str): The ID of the model (e.g., "llama3.1-8b").
        sae_id (str): The ID of the SAE (e.g., "0-llamascope-res-32k").
        output_path (str): The file path to save the resulting merged CSV.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

    # Use anonymous access for the public bucket
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Construct the prefix to list the explanation files
    s3_prefix = f"v1/{model_id}/{sae_id}/explanations/"
    
    logger.info(f"Listing files from s3://{S3_BUCKET_NAME}/{s3_prefix}")

    try:
        # Find all .jsonl.gz files in the specified S3 directory
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix)
        explanation_files = [
            obj['Key'] for page in pages for obj in page.get('Contents', []) 
            if obj['Key'].endswith('.jsonl.gz')
        ]

        if not explanation_files:
            logger.error(f"No explanation files found at {s3_prefix}")
            return

        logger.info(f"Found {len(explanation_files)} files to download and merge.")

        all_explanations = []
        # Use tqdm to create a progress bar for processing the files
        with tqdm(total=len(explanation_files), desc=f"Processing files for {model_id}/{sae_id}") as pbar:
            for file_key in explanation_files:
                s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
                
                # Decompress the gzipped content
                gzipped_content = s3_object['Body'].read()
                decompressed_content = gzip.decompress(gzipped_content)
                
                # Decode and parse each line as a separate JSON object (JSONL format)
                jsonl_content = decompressed_content.decode('utf-8').strip()
                for line in jsonl_content.split('\n'):
                    if line:
                        all_explanations.append(json.loads(line))
                
                pbar.update(1)

        logger.info(f"Successfully processed all files. Total explanations: {len(all_explanations)}")

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(all_explanations)

        # Save the DataFrame to a single CSV file
        df.to_csv(output_path, index=False)
        logger.info(f"Merged data successfully saved to {output_path}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
   #if dir not exist, create it
    # Ensure the output directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    safe_sae_id = SAE_ID.replace('/', '_')
    OUTPUT_FILE = f"data/explanations_{MODEL_ID}_{safe_sae_id}.csv"

    logger.info("Starting S3 explanation downloader script.")
    download_explanations_from_s3(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        output_path=OUTPUT_FILE
    )
    logger.info("Script finished.")