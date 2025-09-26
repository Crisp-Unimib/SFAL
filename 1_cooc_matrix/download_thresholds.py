import requests
import json
import os
from tqdm import tqdm
import concurrent.futures
import time
import random

# --- Configuration ---
# You can change these values to query different models, sources, or feature ranges.
MODEL_ID = "llama3.1-8b"
SOURCE = "31-llamascope-res-32k"
START_INDEX = 0
END_INDEX = 32_768 # The script will query indices up to (but not including) this number.
MAX_WORKERS = 20 # Number of parallel requests to make.
MAX_RETRIES = 100  # Maximum number of retries for a single request.

# Dynamically create the output filename based on model and source.
# This replaces characters that might be invalid in filenames.
sanitized_source = SOURCE.replace("/", "_").replace("-", "_")
OUTPUT_FILENAME = f"thresholds/{MODEL_ID}_{sanitized_source}_results.json"

# API endpoint URL
API_URL = "https://www.neuronpedia.org/api/activation/get"

# Standard headers for the request
HEADERS = {
    "Content-Type": "application/json"
}

def get_last_bin_max_for_feature(model_id, source, index):
    """
    Sends a request to the Neuronpedia API for a specific feature index
    and returns the 'binMax' value from the last item in the response list.
    Includes retry logic with exponential backoff for 429 errors.

    Args:
        model_id (str): The ID of the model to query.
        source (str): The source layer/component of the model.
        index (int): The feature index to query.

    Returns:
        float: The last 'binMax' value if found, otherwise None.
    """
    # The API expects the index as a string in the JSON payload.
    payload = {
        "modelId": model_id,
        "source": source,
        "index": str(index)
    }

    base_backoff = 1  # Base backoff time in seconds

    for attempt in range(MAX_RETRIES):
        try:
            # Send the POST request to the API
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=15)

            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            # Check if the response data is a list and is not empty
            if isinstance(data, list) and data:
                # Get the last dictionary (interval) from the list
                last_interval = data[-1]
                
                # Extract the 'binMax' value from the last interval
                if 'binMax' in last_interval:
                    return last_interval['binMax']
            # If successful but data is not as expected, return None and don't retry.
            return None

        except requests.exceptions.HTTPError as http_err:
            # Check if the error is due to rate limiting (429)
            if http_err.response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    # Calculate exponential backoff time with random jitter
                    backoff_time = base_backoff * (2 ** attempt) + random.uniform(0, 1)
                    tqdm.write(f"  - Index {index}: Rate limit hit. Retrying in {backoff_time:.2f}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(backoff_time)
                else:
                    tqdm.write(f"  - Index {index}: Max retries reached for rate limit. Giving up.")
                    break # Exit retry loop
            else:
                # For other HTTP errors, log and do not retry.
                tqdm.write(f"  - Unrecoverable HTTP error for index {index}: {http_err}")
                break # Exit retry loop
        except requests.exceptions.RequestException as req_err:
            tqdm.write(f"  - Request error occurred for index {index}: {req_err}")
            break # Exit retry loop
        except json.JSONDecodeError:
            tqdm.write(f"  - Failed to decode JSON response for index {index}.")
            break # Exit retry loop
    
    return None

def main():
    """
    Main function to iterate through feature indices in parallel, print results,
    and save them to a JSON file.
    """
    print(f"Starting query for model '{MODEL_ID}' and source '{SOURCE}'.")
    print(f"Querying indices from {START_INDEX} to {END_INDEX - 1} using up to {MAX_WORKERS} workers.")
    print(f"Results will be saved to: {OUTPUT_FILENAME}\n")

    # Dictionary to store the results
    results = {}
    indices_to_query = range(START_INDEX, END_INDEX)

    # Use ThreadPoolExecutor to make requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map futures to their index
        future_to_index = {executor.submit(get_last_bin_max_for_feature, MODEL_ID, SOURCE, i): i for i in indices_to_query}

        # Create a progress bar for the completed futures
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(indices_to_query), desc="Fetching features", unit="feature"):
            index = future_to_index[future]
            try:
                last_bin_max = future.result()
                if last_bin_max is not None:
                    # Store results with the original index as the key
                    results[index] = last_bin_max
            except Exception as exc:
                tqdm.write(f"  - Index {index} generated an exception: {exc}")

    # Print a summary of the results
    print("\n--- Query Complete ---")
    if results:
        # Sort the results by index for consistent output
        sorted_results = dict(sorted(results.items()))

        print("Successfully retrieved values:")
        for index, value in sorted_results.items():
            print(f"  Index: {index}, Last binMax: {value}")

        # Save the results dictionary to a JSON file
        try:
            with open(OUTPUT_FILENAME, 'w') as f:
                # Use indent=4 for a nicely formatted JSON file
                json.dump(sorted_results, f, indent=4)
            print(f"\nSuccessfully saved results to '{OUTPUT_FILENAME}'")
        except IOError as e:
            print(f"\nError: Could not write to file '{OUTPUT_FILENAME}'. Reason: {e}")

    else:
        print("No data was retrieved.")

if __name__ == "__main__":
    main()
