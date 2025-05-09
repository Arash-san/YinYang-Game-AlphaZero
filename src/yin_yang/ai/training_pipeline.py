import os
import numpy as np
import torch
import glob
import time
import logging
import random
from collections import deque
from typing import List, Dict, Tuple, Optional

from .neural_network import YinYangNeuralNetwork
from .trainer import AlphaZeroTrainer
from .data_utils import create_dataset_from_games

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YinYangTraining')

class TrainingDataQueue:
    """
    Queue for training data management.
    """
    def __init__(self, max_size=500000, sample_size=10000):
        """
        Initialize the training data queue.
        
        Args:
            max_size: Maximum number of examples in the queue
            sample_size: Number of examples to sample for each training batch
        """
        self.max_size = max_size
        self.sample_size = min(sample_size, max_size)
        self.queue = deque(maxlen=max_size)
        logger.info(f"TrainingDataQueue initialized with max_size={max_size}, sample_size={sample_size}")
        
    def push_examples(self, examples: List[Tuple]):
        """
        Push new examples to the queue.
        
        Args:
            examples: List of (board, policy, value) tuples
        """
        old_size = len(self.queue)
        
        # Add new examples
        for example in examples:
            self.queue.append(example)
            
        new_size = len(self.queue)
        logger.info(f"Added {new_size - old_size} examples to queue. Queue size: {new_size}")
        
    def push_file(self, file_path: str):
        """
        Push examples from a file to the queue.
        
        Args:
            file_path: Path to the .npz file with examples
        """
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found")
            return
            
        # Load examples from file
        data = np.load(file_path, allow_pickle=True)
        boards = data['boards']
        policies = data['policies']
        values = data['values']
        
        examples = [(boards[i], policies[i], values[i]) for i in range(len(boards))]
        
        # Push examples to queue
        self.push_examples(examples)
        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        
    def sample(self, sample_size=None):
        """
        Sample examples from the queue.
        
        Args:
            sample_size: Number of examples to sample
            
        Returns:
            examples: List of (board, policy, value) tuples
        """
        if sample_size is None:
            sample_size = self.sample_size
            
        # If queue is empty, return empty list
        if len(self.queue) == 0:
            return []
            
        # If sample size is larger than queue size, use queue size
        sample_size = min(sample_size, len(self.queue))
        
        # Sample examples
        samples = random.sample(list(self.queue), sample_size)
        logger.info(f"Sampled {len(samples)} examples from queue of size {len(self.queue)}")
        
        return samples
        
    def __len__(self):
        return len(self.queue)

class TrainingPipeline:
    """
    Pipeline for training the AlphaZero model.
    """
    def __init__(self, game, model_dir='models', data_dir='data', 
                 lr=0.001, batch_size=64, weight_decay=1e-4,
                 epochs_per_iteration=10, sample_size=10000, 
                 queue_size=500000, checkpoint_interval=10):
        """
        Initialize the training pipeline.
        
        Args:
            game: Game instance
            model_dir: Directory to save/load models
            data_dir: Directory with training data
            lr: Learning rate
            batch_size: Batch size for training
            weight_decay: L2 regularization parameter
            epochs_per_iteration: Number of epochs per training iteration
            sample_size: Number of examples to sample for each training iteration
            queue_size: Maximum number of examples in the queue
            checkpoint_interval: Number of iterations between checkpoints
        """
        self.game = game
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs_per_iteration = epochs_per_iteration
        self.sample_size = sample_size
        self.checkpoint_interval = checkpoint_interval
        
        # Create directories if they don't exist
        for directory in [model_dir, data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Create trainer
        self.trainer = AlphaZeroTrainer(
            game=game,
            model_dir=model_dir,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay
        )
        
        # Create data queue
        self.data_queue = TrainingDataQueue(
            max_size=queue_size,
            sample_size=sample_size
        )
        
        # Track iterations
        self.iteration = 0
        
        # Load iteration from last checkpoint if exists
        self._load_iteration()
        
        logger.info(f"TrainingPipeline initialized with model_dir={model_dir}, data_dir={data_dir}")
        logger.info(f"Training parameters: lr={lr}, batch_size={batch_size}, "
                   f"weight_decay={weight_decay}, epochs_per_iteration={epochs_per_iteration}")
        
    def _load_iteration(self):
        """Load the iteration number from the latest checkpoint."""
        # Find all checkpoints
        checkpoint_pattern = os.path.join(self.model_dir, "checkpoint_*.pth.tar")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            logger.info("No checkpoints found. Starting from iteration 0.")
            return
            
        # Extract iteration numbers from checkpoint files
        iterations = [int(os.path.basename(cp).split('_')[1].split('.')[0]) 
                     for cp in checkpoints]
        latest_iteration = max(iterations)
        
        # Load the latest checkpoint
        self.trainer.load_checkpoint(iteration=latest_iteration)
        self.iteration = latest_iteration
        
        logger.info(f"Loaded model from iteration {latest_iteration}")
        
    def load_data(self):
        """Load training data from files in the data directory."""
        # Find all data files
        data_pattern = os.path.join(self.data_dir, "self_play_data_*.npz")
        data_files = glob.glob(data_pattern)
        
        if not data_files:
            logger.warning("No data files found.")
            return
            
        # Load data from files
        for file_path in data_files:
            self.data_queue.push_file(file_path)
            
        logger.info(f"Loaded {len(data_files)} data files with {len(self.data_queue)} examples total")
        
    def train_iteration(self):
        """Run one iteration of training."""
        # Sample examples from the queue
        examples = self.data_queue.sample()
        
        if not examples:
            logger.warning("No examples available for training.")
            return {}
            
        # Train the model
        metrics = self.trainer.train(
            examples=examples,
            epochs=self.epochs_per_iteration,
            augment=True
        )
        
        # Increment iteration
        self.iteration += 1
        
        # Save checkpoint
        if self.iteration % self.checkpoint_interval == 0:
            self.trainer.save_checkpoint(iteration=self.iteration)
            logger.info(f"Saved checkpoint at iteration {self.iteration}")
            
        # Log metrics
        logger.info(f"Iteration {self.iteration} completed. "
                   f"Policy Loss: {metrics['policy_loss'][-1]:.4f}, "
                   f"Value Loss: {metrics['value_loss'][-1]:.4f}, "
                   f"Total Loss: {metrics['total_loss'][-1]:.4f}")
        
        return metrics
        
    def train(self, num_iterations=10):
        """
        Run multiple iterations of training.
        
        Args:
            num_iterations: Number of training iterations to run
        
        Returns:
            metrics: Dictionary of training metrics
        """
        all_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        for i in range(num_iterations):
            logger.info(f"Starting training iteration {i+1}/{num_iterations}")
            
            # Run one iteration of training
            metrics = self.train_iteration()
            
            # Collect metrics
            if metrics:
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].extend(metrics[key])
            
            # Log progress
            logger.info(f"Completed training iteration {i+1}/{num_iterations}")
            
        return all_metrics
        
    def get_latest_model_path(self):
        """Get the path to the latest model."""
        if self.iteration > 0:
            return os.path.join(self.model_dir, f"checkpoint_{self.iteration}.pth.tar")
        else:
            return os.path.join(self.model_dir, "checkpoint.pth.tar")
            
    def get_model_performance(self, model_path=None):
        """
        Evaluate model performance on validation data.
        
        Args:
            model_path: Path to the model to evaluate
            
        Returns:
            metrics: Dictionary of validation metrics
        """
        # TODO: Implement model evaluation
        pass

def run_training_pipeline(game, model_dir='models', data_dir='data', 
                         num_iterations=10, sample_size=10000, checkpoint_interval=10):
    """
    Run the complete training pipeline.
    
    Args:
        game: Game instance
        model_dir: Directory to save/load models
        data_dir: Directory with training data
        num_iterations: Number of training iterations to run
        sample_size: Number of examples to sample for each training iteration
        checkpoint_interval: Number of iterations between checkpoints
    """
    # Create pipeline
    pipeline = TrainingPipeline(
        game=game,
        model_dir=model_dir,
        data_dir=data_dir,
        sample_size=sample_size,
        checkpoint_interval=checkpoint_interval
    )
    
    # Load data
    pipeline.load_data()
    
    # Train
    metrics = pipeline.train(num_iterations=num_iterations)
    
    # Log final results
    if metrics['total_loss']:
        final_loss = metrics['total_loss'][-1]
        logger.info(f"Training completed. Final loss: {final_loss:.4f}")
    else:
        logger.warning("Training completed, but no metrics were recorded.")
        
    # Return latest model path
    return pipeline.get_latest_model_path() 