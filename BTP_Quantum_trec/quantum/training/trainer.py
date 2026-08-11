import torch
import torch.nn as nn
import torch.optim as optim

class MetaLearningTrainer:
    """
    Handles the inner loop of the training process (VQC Parameter Tuning).
    Uses PyTorch's Adam optimizer to calculate quantum gradients via the TorchConnector 
    and backpropagate the loss.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        """
        Initialize the trainer.
        Args:
            model (nn.Module): The QuantumPrototypicalNetwork.
            learning_rate (float): The step size for the optimizer.
        """
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # In a real prototypical network, loss is calculated manually via negative log-likelihood 
        # over the quantum infidelities (Blocks 4, 5, 6). 
        # Here we mock a standard CrossEntropyLoss for structural completeness.
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                   query_x: torch.Tensor, query_y: torch.Tensor) -> float:
        """
        Executes a single episodic training step (Forward pass + Backprop).
        
        Args:
            support_x: Features of the support set.
            support_y: Labels of the support set.
            query_x: Features of the query set.
            query_y: Labels of the query set.
            
        Returns:
            float: The loss value for this step.
        """
        self.model.train()
        
        # 1. Zero the gradients
        self.optimizer.zero_grad()
        
        # 2. Forward pass (Through Angle Encoding -> VQC -> Measurement)
        # Note: In a full QPN, you would calculate the prototypes from support_x here,
        # calculate infidelity distances from query_x to prototypes, and apply softmax.
        # This is a simplified structural representation.
        predictions = self.model(query_x)
        
        # 3. Calculate Loss (Block 8)
        loss = self.loss_fn(predictions, query_y)
        
        # 4. Backward pass (Quantum Gradients via Parameter-Shift Rule)
        loss.backward()
        
        # 5. Update VQC Angles
        self.optimizer.step()
        
        return loss.item()
