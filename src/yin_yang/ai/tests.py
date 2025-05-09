import unittest
import torch
import numpy as np
import os
import logging
import sys
import tempfile
import shutil

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add logging handler to output to console for tests
logger = logging.getLogger('YinYangNN')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Import the necessary modules
from src.yin_yang import YinYangGame
from src.yin_yang.ai.neural_network import YinYangNeuralNetwork
from src.yin_yang.ai.data_utils import DataProcessor, create_dataset_from_games
from src.yin_yang.ai.trainer import AlphaZeroTrainer
from src.yin_yang.ai.self_play import SelfPlayWorker
from src.yin_yang.ai.training_pipeline import TrainingDataQueue, TrainingPipeline

class TestNeuralNetwork(unittest.TestCase):
    """
    Test cases for the YinYangNeuralNetwork class.
    """
    def setUp(self):
        """Set up test fixtures."""
        self.game = YinYangGame(n=6, m=6)  # Smaller board for faster tests
        self.net = YinYangNeuralNetwork(self.game, num_channels=32, num_res_blocks=2)
        
    def test_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.net.board_size, (6, 6))
        self.assertEqual(self.net.action_size, 36)  # 6x6 board
        self.assertEqual(self.net.input_channels, 5)  # 3 piece channels + 2 feature channels
        
    def test_forward_pass(self):
        """Test forward pass through the network."""
        # Create a batch of 2 random board states
        x = torch.rand(2, 5, 6, 6)
        
        # Forward pass
        policy_logits, value = self.net(x)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (2, 36))  # (batch_size, action_size)
        self.assertEqual(value.shape, (2, 1))  # (batch_size, 1)
        
        # Check value range
        self.assertTrue(torch.all(value >= -1) and torch.all(value <= 1))
        
    def test_predict(self):
        """Test prediction on a game board."""
        # Create a board
        board = self.game.getInitBoard()
        
        # Make some moves
        board, _ = self.game.getNextState(board, 1, 0)  # Place black at (0,0)
        board, _ = self.game.getNextState(board, -1, 7)  # Place white at (1,1)
        
        # Get prediction
        policy, value = self.net.predict(board)
        
        # Check output shapes
        self.assertEqual(policy.shape, (36,))  # action_size
        self.assertTrue(isinstance(value, float))
        
        # Check policy is a probability distribution
        self.assertAlmostEqual(np.sum(policy), 1.0, places=6)
        self.assertTrue(np.all(policy >= 0) and np.all(policy <= 1))
        
    def test_board_to_input(self):
        """Test conversion of board to network input."""
        # Create a board
        board = self.game.getInitBoard()
        
        # Make some moves
        board, _ = self.game.getNextState(board, 1, 0)  # Place black at (0,0)
        board, _ = self.game.getNextState(board, -1, 7)  # Place white at (1,1)
        
        # Convert to input
        x = self.net.board_to_input(board)
        
        # Check shape
        self.assertEqual(x.shape, (5, 6, 6))  # (channels, height, width)
        
        # Check channel values
        self.assertEqual(x[0, 0, 0], 0)  # Not empty at (0,0)
        self.assertEqual(x[1, 0, 0], 1)  # Black piece at (0,0)
        self.assertEqual(x[2, 0, 0], 0)  # No white piece at (0,0)
        
        self.assertEqual(x[0, 1, 1], 0)  # Not empty at (1,1)
        self.assertEqual(x[1, 1, 1], 0)  # No black piece at (1,1)
        self.assertEqual(x[2, 1, 1], 1)  # White piece at (1,1)
        
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth.tar', delete=False) as f:
            model_path = f.name
            
        try:
            # Save the model
            self.net.save_model(model_path)
            
            # Create a new network
            new_net = YinYangNeuralNetwork(self.game, num_channels=32, num_res_blocks=2)
            
            # Load the model
            new_net.load_model(model_path)
            
            # Check that the models have the same weights
            for p1, p2 in zip(self.net.parameters(), new_net.parameters()):
                self.assertTrue(torch.all(torch.isclose(p1, p2)))
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
                
    def test_train(self):
        """Test training on synthetic data."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create synthetic training data
            boards = []
            policies = []
            values = []
            
            # Add 10 random examples
            for _ in range(10):
                # Random board
                board = self.game.getInitBoard()
                
                # Random policy
                policy = np.random.rand(36)
                policy = policy / np.sum(policy)
                
                # Random value
                value = np.random.uniform(-1, 1)
                
                boards.append(board)
                policies.append(policy)
                values.append(value)
            
            # Create dataset
            examples = [(boards[i], policies[i], values[i]) for i in range(10)]
            
            # Create trainer
            trainer = AlphaZeroTrainer(
                game=self.game,
                model_dir=temp_dir,
                lr=0.01,
                batch_size=5
            )
            
            # Train for 1 epoch
            metrics = trainer.train(examples, epochs=1)
            
            # Check that metrics are recorded
            self.assertIn('policy_loss', metrics)
            self.assertIn('value_loss', metrics)
            self.assertIn('total_loss', metrics)
            
            # Check that losses are reasonable
            self.assertTrue(all(l > 0 for l in metrics['total_loss']))
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

class TestSelfPlay(unittest.TestCase):
    """
    Test cases for self-play functionality.
    """
    def setUp(self):
        """Set up test fixtures."""
        self.game = YinYangGame(n=4, m=4)  # Smaller board for faster tests
        
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth.tar')
        
        # Create a test model
        self.net = YinYangNeuralNetwork(self.game, num_channels=16, num_res_blocks=1)
        self.net.save_model(self.model_path)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
    def test_self_play_worker(self):
        """Test that a self-play worker can generate games."""
        # Create a worker with minimal computational requirements
        worker = SelfPlayWorker(
            game=self.game,
            model_path=self.model_path,
            num_simulations=5,  # Very few simulations for testing
            num_games=1,
            num_parallel=1
        )
        
        # Generate one game
        examples = worker.play_game()
        
        # Check that examples were generated
        self.assertTrue(len(examples) > 0)
        
        # Check the structure of examples
        for board, policy, value in examples:
            # Board should be a valid board state
            self.assertEqual(board.board.shape, (4, 4))
            
            # Policy should be a probability distribution
            self.assertEqual(policy.shape, (16,))
            self.assertAlmostEqual(np.sum(policy), 1.0, places=6)
            self.assertTrue(np.all(policy >= 0) and np.all(policy <= 1))
            
            # Value should be -1, 0, or 1
            self.assertTrue(-1 <= value <= 1)

class TestTrainingPipeline(unittest.TestCase):
    """
    Test cases for the training pipeline.
    """
    def setUp(self):
        """Set up test fixtures."""
        self.game = YinYangGame(n=4, m=4)  # Smaller board for faster tests
        
        # Create temporary directories
        self.model_dir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        for directory in [self.model_dir, self.data_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
        
    def test_training_data_queue(self):
        """Test the training data queue."""
        # Create a queue
        queue = TrainingDataQueue(max_size=100, sample_size=10)
        
        # Create synthetic examples
        examples = []
        for i in range(20):
            board = self.game.getInitBoard()
            policy = np.ones(16) / 16  # Uniform policy
            value = float(i % 2) * 2 - 1  # Alternating -1 and 1
            examples.append((board, policy, value))
        
        # Push examples to the queue
        queue.push_examples(examples)
        
        # Check queue size
        self.assertEqual(len(queue), 20)
        
        # Sample examples
        samples = queue.sample(5)
        
        # Check sample size
        self.assertEqual(len(samples), 5)
        
        # Check that samples have the correct structure
        for board, policy, value in samples:
            self.assertEqual(board.board.shape, (4, 4))
            self.assertEqual(policy.shape, (16,))
            self.assertTrue(value in [-1, 1])
        
    def test_training_pipeline_initialization(self):
        """Test that the training pipeline initializes correctly."""
        # Create a pipeline
        pipeline = TrainingPipeline(
            game=self.game,
            model_dir=self.model_dir,
            data_dir=self.data_dir,
            epochs_per_iteration=1,
            sample_size=10
        )
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.model_dir))
        self.assertTrue(os.path.exists(self.data_dir))
        
        # Check that the trainer was created
        self.assertIsNotNone(pipeline.trainer)
        
        # Check that the data queue was created
        self.assertIsNotNone(pipeline.data_queue)
        
    def test_single_training_iteration(self):
        """Test a single training iteration with synthetic data."""
        # Create a pipeline
        pipeline = TrainingPipeline(
            game=self.game,
            model_dir=self.model_dir,
            data_dir=self.data_dir,
            epochs_per_iteration=1,
            sample_size=10,
            checkpoint_interval=1  # Save checkpoint every iteration
        )
        
        # Create synthetic examples
        examples = []
        for i in range(20):
            board = self.game.getInitBoard()
            policy = np.ones(16) / 16  # Uniform policy
            value = float(i % 2) * 2 - 1  # Alternating -1 and 1
            examples.append((board, policy, value))
        
        # Push examples to the queue
        pipeline.data_queue.push_examples(examples)
        
        # Run one iteration
        metrics = pipeline.train_iteration()
        
        # Check that metrics were recorded
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)
        self.assertIn('total_loss', metrics)
        
        # Check that iteration was incremented
        self.assertEqual(pipeline.iteration, 1)
        
        # Check that a checkpoint was saved
        checkpoint_path = os.path.join(self.model_dir, "checkpoint_1.pth.tar")
        self.assertTrue(os.path.exists(checkpoint_path))

if __name__ == '__main__':
    unittest.main() 