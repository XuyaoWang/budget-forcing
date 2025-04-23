import os
import glob
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Function to check if a JSON file should be deleted
def check_json_file(json_file):
    try:
        with open(json_file, 'r') as f:
            # data = f.read()
            data = json.load(f)
        if data == "" or data == "Bean Proxy API Failed...":
            return json_file
        return None
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return json_file

# Function to delete a file
def delete_file(file_path):
    try:
        os.remove(file_path)
        return f"Deleted: {file_path}"
    except Exception as e:
        return f"Error deleting {file_path}: {e}"

if __name__ == "__main__":
    DATA_DIR = "./cache"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True)
    json_files = [os.path.abspath(f) for f in json_files]
    
    # Determine number of processes (use 75% of available CPUs)
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} processes")
    
    # Process files in parallel to find which ones to delete
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(check_json_file, json_files),
            total=len(json_files),
            desc="Checking files"
        ))
    
    # Filter out None values
    delete_json_files = [f for f in results if f is not None]
    print(f"Found {len(delete_json_files)} files to delete")
    
    # Delete files in parallel
    with Pool(num_processes) as pool:
        delete_results = list(tqdm(
            pool.imap(delete_file, delete_json_files),
            total=len(delete_json_files),
            desc="Deleting files"
        ))
    
    # Print deletion results
    for result in delete_results:
        print(result)
    
    print("Parallel deletion process completed.")