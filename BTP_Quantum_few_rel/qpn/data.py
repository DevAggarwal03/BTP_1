# This module loads and organizes FewRel embedding data for training and validation splits.
# It turns the saved numpy archives into relation-specific pools that the episode sampler can consume.
import os
import numpy as np
from typing import Dict, Tuple

DATA_DIR = "data"

def load_fewrel_embeddings(split_name: str) -> Dict[str, np.ndarray]:
    """
    Loads precomputed embeddings from data/split_name_embeddings.npz
    Returns a dictionary mapping relation_id -> np.ndarray of shape (num_instances, embedding_dim)
    """
    path = os.path.join(DATA_DIR, f"{split_name}_embeddings.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file {path} not found. Did you run scripts/build_embeddings.py?")
        
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    relation_pool = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        relation_pool[label] = embeddings[mask]
        
    return relation_pool

def get_relation_pools() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns (train_pool, val_pool) where each pool maps relation_id to its instances' embeddings.
    """
    train_pool = load_fewrel_embeddings("train")
    val_pool = load_fewrel_embeddings("val")
    return train_pool, val_pool
