# This script builds and saves sentence embeddings for the FewRel data used by the project.
# It loads text examples, computes embeddings with a transformer model, and writes them into numpy archives.
import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def format_sentence(instance):
    """
    Instance format in FewRel:
    {
        'tokens': ['The', 'city', 'is', ...],
        'h': ['entity1_name', 'id', [[start, end]]],
        't': ['entity2_name', 'id', [[start, end]]]
    }
    Note: FewRel indices [[start, end]] are such that start is inclusive, end is exclusive for the slice. Wait, usually end is the index of the last token.
    Actually, let's look at the standard way: `start` and `end-1` are the token indices.
    We will just insert markers [H] and [/H] around the head entity, [T] and [/T] around the tail entity.
    If there are multiple spans, we just take the first one.
    """
    tokens = instance['tokens'].copy()
    h_span = instance['h'][2][0]
    t_span = instance['t'][2][0]
    
    # We need to insert backwards to not mess up indices
    spans = [
        (h_span[0], h_span[-1], '[H]', '[/H]'),
        (t_span[0], t_span[-1], '[T]', '[/T]')
    ]
    # Sort by start index descending
    spans.sort(key=lambda x: x[0], reverse=True)
    
    for start, end, marker_start, marker_end in spans:
        # Assuming end in FewRel is the last token index (inclusive), so we insert after end + 1
        # Let's assume end is the last token index (inclusive).
        # Actually, in FewRel json: `h[2][0]` is `[pos1, pos2]` where pos1 is start, pos2 is end (both inclusive).
        # Wait, if pos2 is inclusive, we insert after pos2+1.
        # Let's insert at pos2+1 first, then at pos1.
        tokens.insert(end[-1] + 1 if isinstance(end, list) else end + 1, marker_end)
        tokens.insert(start[0] if isinstance(start, list) else start, marker_start)
        
    return " ".join(tokens)

def build_embeddings(split_name):
    json_path = os.path.join(DATA_DIR, f"{split_name}_wiki.json")
    out_path = os.path.join(DATA_DIR, f"{split_name}_embeddings.npz")
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    print(f"Loaded {split_name} data. {len(data)} relations.")
    
    model = SentenceTransformer(MODEL_NAME)
    
    # We will save as a numpy npz file. 
    # Structure: array of embeddings, array of labels (relations), array of instance indices within relation
    embeddings_list = []
    labels_list = []
    
    for relation, instances in tqdm(data.items(), desc=f"Encoding {split_name}"):
        sentences = [format_sentence(inst) for inst in instances]
        
        # Encode in batches
        with torch.no_grad():
            embs = model.encode(sentences, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
            
        embeddings_list.append(embs)
        labels_list.extend([relation] * len(instances))
        
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list)
    
    np.savez(out_path, embeddings=all_embeddings, labels=all_labels)
    print(f"Saved embeddings to {out_path} with shape {all_embeddings.shape}")

def main():
    build_embeddings("train")
    build_embeddings("val")

if __name__ == "__main__":
    main()
