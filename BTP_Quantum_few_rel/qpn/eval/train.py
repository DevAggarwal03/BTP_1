import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from qpn.episodes import EpisodeSampler

def meta_train_qpn(model, train_pool, global_preprocessor, config, n_train_episodes=50, lr=0.05):
    """
    Executes an episodic meta-training loop for the QuantumProtoNet.
    
    model: QuantumProtoNet instance
    train_pool: Dictionary of training relations and sentences
    embeddings: Dictionary of sentence hashes to embedding vectors
    global_preprocessor: Pre-fitted QuantumFeaturePreprocessor instance
    config: EpisodeConfig defining the shape of training episodes
    n_train_episodes: Number of episodes to train on. 50 allows gradient convergence.
    lr: Learning rate for the Adam optimizer.
    """
    model.train() # Enable gradient tracking
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    sampler = EpisodeSampler(train_pool, config)
    episodes = sampler.sample(n_train_episodes)
    
    total_loss = 0.0
    
    for ep_idx, ep in enumerate(episodes):
        optimizer.zero_grad()
        
        # We need to preprocess the episode features globally
        s_x = global_preprocessor.transform(ep.support_x)
        q_x = global_preprocessor.transform(ep.query_x)
        
        # Convert to PyTorch tensors
        s_x_t = torch.tensor(s_x, dtype=torch.float32)
        q_x_t = torch.tensor(q_x, dtype=torch.float32)
        
        # We assume labels are neatly packed into 0..N-1 for s_y_t
        classes = list(np.unique(ep.support_y))
        s_y_mapped = np.array([classes.index(y) for y in ep.support_y])
        s_y_t = torch.tensor(s_y_mapped, dtype=torch.long)
        
        # We map query_y to numerical indices [0..N-1] based on the unique classes in support
        q_y_mapped = np.array([classes.index(y) for y in ep.query_y])
        q_y_t = torch.tensor(q_y_mapped, dtype=torch.long)
        
        # Forward pass through the Differentiable Training Path
        logits = model(s_x_t, q_x_t, s_y_t)
        
        loss = loss_fn(logits, q_y_t)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
        print(f"  [Train] Ep {ep_idx+1}/{n_train_episodes} - Loss: {loss.item():.4f}")
        
    print(f"  => Meta-Training Complete. Avg Loss: {total_loss / n_train_episodes:.4f}")
    return model
