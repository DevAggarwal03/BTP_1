# This module implements a quantum-inspired feature-selection heuristic for choosing useful features.
# It uses a chaotic search procedure to optimize a binary mask over the available input dimensions.
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from .config import QCHBAConfig

class QCHBASelector:
    """
    Quantum Chaotic Honey Badger Algorithm (QCHBA) for Feature Selection.
    Based on "Quantum Chaotic Honey Badger Algorithm for Feature Selection" (MDPI, 2022).
    
    This algorithm implements:
    1. Quantum-inspired initialization (using Q-bits).
    2. Chaotic maps (Logistic map) for parameter tuning.
    3. Honey Badger foraging behaviors (Mining and Honey-seeking phases).
    
    NOTE: This feature selector is strictly **quantum-inspired**, runs entirely on classical CPU,
    and has absolutely no qubits.
    """
    def __init__(self, config: QCHBAConfig, pop_size=20, max_iter=50):
        self.config = config
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.selected_indices = None
        
    def _logistic_map(self, iter_idx, num_vals):
        """Generates chaotic sequence using Logistic Map."""
        # Simple logistic map: x_{t+1} = 4 * x_t * (1 - x_t)
        # Using a fixed seed based on iteration for chaos
        np.random.seed(iter_idx)
        chaos = np.zeros(num_vals)
        x = np.random.rand()
        for i in range(num_vals):
            x = 4.0 * x * (1.0 - x)
            chaos[i] = x
        return chaos

    def _fitness(self, position, X, y):
        """
        Evaluates the fitness of a binary position vector.
        Fitness = alpha * ErrorRate + (1 - alpha) * (SelectedFeatures / TotalFeatures)
        """
        # Convert to boolean mask
        mask = position > 0.5
        if not np.any(mask):
            return 1.0 # Worst fitness if no features selected
            
        X_subset = X[:, mask]
        
        # Use a simple kNN for fast fitness evaluation
        knn = KNeighborsClassifier(n_neighbors=3)
        # 3-fold CV
        scores = cross_val_score(knn, X_subset, y, cv=3)
        error_rate = 1.0 - np.mean(scores)
        
        alpha = 0.99
        feature_ratio = np.sum(mask) / len(mask)
        
        fitness = alpha * error_rate + (1.0 - alpha) * feature_ratio
        return fitness

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the QCHBA to select the most discriminative features.
        X: (N, D)
        y: (N,)
        """
        num_features = X.shape[1]
        
        print(f"  [QCHBA] Starting fit over {self.max_iter} iterations (pop_size={self.pop_size})...")
        
        # 1. Quantum-Inspired Initialization
        # Initialize angles for Q-bits in [0, pi/2]
        np.random.seed(42) # Ensure reproducibility
        angles = np.random.rand(self.pop_size, num_features) * (np.pi / 2)
        
        # Positions are determined by measuring Q-bits (probability = sin^2(theta))
        probs = np.sin(angles) ** 2
        positions = (np.random.rand(self.pop_size, num_features) < probs).astype(float)
        
        fitness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness_values[i] = self._fitness(positions[i], X, y)
            
        best_idx = np.argmin(fitness_values)
        global_best_pos = positions[best_idx].copy()
        global_best_fit = fitness_values[best_idx]
        
        # QCHBA Main Loop
        C = 2.0 # Constant for HBA
        
        for t in range(self.max_iter):
            # Alpha parameter decreases over time
            alpha = C * np.exp(-t / self.max_iter)
            
            # Chaotic maps for randomness
            chaos_r1 = self._logistic_map(t * 3, self.pop_size)
            chaos_r2 = self._logistic_map(t * 3 + 1, self.pop_size)
            chaos_r3 = self._logistic_map(t * 3 + 2, self.pop_size)
            
            for i in range(self.pop_size):
                # Update smell intensity (I)
                di = np.linalg.norm(positions[i] - global_best_pos) + 1e-6
                S = chaos_r1[i] # Source strength (chaotic)
                I = chaos_r2[i] * S / (4 * np.pi * di ** 2)
                
                # Update position based on phase
                if chaos_r3[i] < 0.5:
                    # Mining phase
                    F = 1 if np.random.rand() < 0.5 else -1
                    new_pos = global_best_pos + F * alpha * I * global_best_pos + F * chaos_r1[i] * alpha * di
                else:
                    # Honey-seeking phase
                    new_pos = global_best_pos + chaos_r2[i] * alpha * di
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_pos_prob = 1.0 / (1.0 + np.exp(-new_pos))
                new_pos_bin = (np.random.rand(num_features) < new_pos_prob).astype(float)
                
                new_fit = self._fitness(new_pos_bin, X, y)
                
                if new_fit < fitness_values[i]:
                    positions[i] = new_pos_bin
                    fitness_values[i] = new_fit
                    
                    if new_fit < global_best_fit:
                        global_best_fit = new_fit
                        global_best_pos = new_pos_bin.copy()
                        
            if (t + 1) % 5 == 0 or t == self.max_iter - 1:
                print(f"  [QCHBA] Iteration {t+1}/{self.max_iter} | Best Fitness: {global_best_fit:.4f} | Features Masked: {np.sum(global_best_pos > 0.5)}/{num_features}")
                        
        # Extract the selected indices
        self.selected_indices = np.where(global_best_pos > 0.5)[0]
        print(f"  [QCHBA] Done! Selected {len(self.selected_indices)} optimal features out of {num_features}.")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduces X to the selected features.
        X: (..., D)
        """
        if self.selected_indices is None:
            raise ValueError("QCHBASelector is not fitted yet.")
        return X[..., self.selected_indices]
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
