# This script downloads the FewRel train and validation datasets used for experiments.
# It fetches the JSON files from the upstream repository into the data directory.
import os
import urllib.request
import json

TRAIN_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json"
VAL_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json"

DATA_DIR = "data"

def download_file(url: str, dest: str):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    train_path = os.path.join(DATA_DIR, "train_wiki.json")
    val_path = os.path.join(DATA_DIR, "val_wiki.json")
    
    if not os.path.exists(train_path):
        download_file(TRAIN_URL, train_path)
    else:
        print(f"{train_path} already exists.")
        
    if not os.path.exists(val_path):
        download_file(VAL_URL, val_path)
    else:
        print(f"{val_path} already exists.")
        
    # Verify JSON
    with open(train_path, "r") as f:
        train_data = json.load(f)
        print(f"Loaded train data with {len(train_data)} relations.")
        
    with open(val_path, "r") as f:
        val_data = json.load(f)
        print(f"Loaded val data with {len(val_data)} relations.")

if __name__ == "__main__":
    main()
