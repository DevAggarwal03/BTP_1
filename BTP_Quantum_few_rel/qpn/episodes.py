# This module defines episode objects and the sampler used to generate few-shot tasks.
# It remaps classes to integer labels and constructs support/query examples from relation pools.
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .config import EpisodeConfig

@dataclass
class Episode:
    """
    Represents a single few-shot episode.
    Classes are remapped to 0, 1, ..., N-1.
    """
    support_x: np.ndarray # Shape: (N * K, D)
    support_y: np.ndarray # Shape: (N * K,)
    query_x: np.ndarray   # Shape: (N * Q, D)
    query_y: np.ndarray   # Shape: (N * Q,)
    classes: List[str]    # Original relation IDs (length N)


class EpisodeSampler:
    def __init__(self, relation_pool: Dict[str, np.ndarray], config: EpisodeConfig):
        """
        relation_pool: Dictionary mapping relation string -> np.ndarray of features for all instances
        config: N-way, K-shot, Q-queries
        """
        self.relation_pool = relation_pool
        self.config = config
        self.relations = list(relation_pool.keys())
        
        # Ensure we have enough relations for N-way
        if len(self.relations) < config.n_way:
            raise ValueError(f"Pool only has {len(self.relations)} relations, but {config.n_way}-way is requested.")

    def _sample_episode(self) -> Episode:
        # 1. Sample N classes without replacement
        selected_classes = random.sample(self.relations, self.config.n_way)
        
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        
        for class_idx, relation in enumerate(selected_classes):
            instances = self.relation_pool[relation]
            num_instances = len(instances)
            req_instances = self.config.k_shot + self.config.q_queries
            
            if num_instances < req_instances:
                raise ValueError(f"Relation {relation} only has {num_instances} instances, "
                                 f"but {req_instances} are needed (K={self.config.k_shot}, Q={self.config.q_queries}).")
            
            # Sample K+Q indices
            selected_indices = random.sample(range(num_instances), req_instances)
            support_indices = selected_indices[:self.config.k_shot]
            query_indices = selected_indices[self.config.k_shot:]
            
            support_x_list.append(instances[support_indices])
            support_y_list.extend([class_idx] * self.config.k_shot)
            
            query_x_list.append(instances[query_indices])
            query_y_list.extend([class_idx] * self.config.q_queries)
            
        support_x = np.concatenate(support_x_list, axis=0)
        support_y = np.array(support_y_list)
        
        query_x = np.concatenate(query_x_list, axis=0)
        query_y = np.array(query_y_list)
        
        return Episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            classes=selected_classes
        )

    def sample(self, n_episodes: int = None) -> List[Episode]:
        """
        Returns a list of sampled episodes.
        """
        if n_episodes is None:
            n_episodes = self.config.n_episodes
            
        return [self._sample_episode() for _ in range(n_episodes)]
