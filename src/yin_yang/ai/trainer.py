import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import logging
import os

from .neural_network import YinYangNeuralNetwork
from .data_utils import create_dataset_from_games

logger = logging.getLogger('YinYangNN.trainer')

class AlphaZeroTrainer:
    """
    Trainer for the AlphaZero-style neural network.
    """
    def __init__(self, game, model_dir='models', 
                 lr=0.001, batch_size=64, weight_decay=1e-4,
                 device=None):
        """
        Initialize the trainer.
        
        Args:
            game: Game instance
            model_dir: Directory to save/load models
            lr: Learning rate
            batch_size: Batch size for training
            weight_decay: L2 regularization parameter
            device: Device to use for training (None for auto-detection)
        """
        self.game = game
        self.model_dir = model_dir
        self.batch_size = batch_size
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Training on device: {self.device}")
        
        # Create neural network
        self.nnet = YinYangNeuralNetwork(game)
        self.nnet.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.nnet.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        
        # Log training parameters
        logger.info(f"Training parameters: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
        
    def train(self, examples, epochs=10, augment=True):
        """
        Train the neural network on the provided examples.
        
        Args:
            examples: List of training examples (board, policy, value)
            epochs: Number of epochs to train
            augment: Whether to use data augmentation
            
        Returns:
            metrics: Dictionary of training metrics
        """
        logger.info(f"Training on {len(examples)} examples for {epochs} epochs")
        
        # Create dataset from examples
        board_tensors, policy_tensors, value_tensors = create_dataset_from_games(
            examples, self.game, augment=augment
        )
        
        # Convert to PyTorch tensors
        board_tensors = torch.stack(board_tensors)
        
        # Ensure all policy tensors are PyTorch tensors, not numpy arrays
        policy_tensors = [torch.tensor(p) if isinstance(p, np.ndarray) else p for p in policy_tensors]
        policy_tensors = torch.stack(policy_tensors)
        
        value_tensors = torch.cat(value_tensors)
        
        # Create dataset and dataloader
        dataset = TensorDataset(board_tensors, policy_tensors, value_tensors)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training metrics
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        # Training loop
        self.nnet.train()
        for epoch in range(epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            
            start_time = time.time()
            
            for batch_idx, (boards, policies, values) in enumerate(dataloader):
                # Move data to device
                boards = boards.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value_preds = self.nnet(boards)
                
                # Calculate loss
                policy_loss = self.policy_loss_fn(policy_logits, policies)
                value_loss = self.value_loss_fn(value_preds.view(-1), values)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_policy_loss += policy_loss.item() * boards.size(0)
                epoch_value_loss += value_loss.item() * boards.size(0)
                epoch_total_loss += total_loss.item() * boards.size(0)
                
            # Calculate epoch metrics
            epoch_policy_loss /= len(dataset)
            epoch_value_loss /= len(dataset)
            epoch_total_loss /= len(dataset)
            
            metrics['policy_loss'].append(epoch_policy_loss)
            metrics['value_loss'].append(epoch_value_loss)
            metrics['total_loss'].append(epoch_total_loss)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Policy Loss: {epoch_policy_loss:.4f}, "
                       f"Value Loss: {epoch_value_loss:.4f}, "
                       f"Total Loss: {epoch_total_loss:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        
        return metrics
    
    def save_checkpoint(self, filename=None, iteration=None):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the file to save the model (default: checkpoint.pth.tar)
            iteration: Current iteration number (for filename)
        """
        if filename is None:
            if iteration is not None:
                filename = f"checkpoint_{iteration}.pth.tar"
            else:
                filename = "checkpoint.pth.tar"
                
        filepath = os.path.join(self.model_dir, filename)
        self.nnet.save_model(filepath)
        
    def load_checkpoint(self, filename=None, iteration=None):
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the file to load the model from (default: checkpoint.pth.tar)
            iteration: Iteration number to load (for filename)
        """
        if filename is None:
            if iteration is not None:
                filename = f"checkpoint_{iteration}.pth.tar"
            else:
                filename = "checkpoint.pth.tar"
                
        filepath = os.path.join(self.model_dir, filename)
        
        if os.path.exists(filepath):
            self.nnet.load_model(filepath)
            logger.info(f"Loaded model from {filepath}")
        else:
            logger.warning(f"No checkpoint found at {filepath}")
            
    def predict(self, board):
        """
        Make a prediction for the given board.
        
        Args:
            board: Board state
            
        Returns:
            policy: Policy output (probability distribution over actions)
            value: Value output (evaluation of current position)
        """
        return self.nnet.predict(board) 