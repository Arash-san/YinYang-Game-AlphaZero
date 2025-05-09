import unittest
import numpy as np
import os
import sys
import logging
import time
import math
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable logger output during tests
logging.disable(logging.CRITICAL)

from src.yin_yang import YinYangGame
from src.yin_yang.ai.mcts import Node, MCTS
from src.yin_yang.ai.neural_network import YinYangNeuralNetwork

class MockNeuralNetwork:
    """Mock neural network for testing."""
    def __init__(self, policy=None, value=None):
        self.policy = policy
        self.value = value
        self.predict_calls = 0
        
    def predict(self, board):
        """Return predetermined policy and value."""
        self.predict_calls += 1
        return self.policy, self.value

class SimpleBoard:
    """Simple board state for testing."""
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        
    def get_board(self):
        return self.board

class SimpleGame:
    """Simple game for testing."""
    def __init__(self, size=3, valid_moves=None, game_result=0):
        self.size = size
        self.action_size = size * size
        self._valid_moves = valid_moves
        self._game_result = game_result
        
    def getActionSize(self):
        return self.action_size
        
    def getValidMoves(self, board, player):
        if self._valid_moves is not None:
            return self._valid_moves
        # All moves are valid by default
        return np.ones(self.action_size)
        
    def getGameEnded(self, board, player):
        return self._game_result
        
    def getNextState(self, board, player, action):
        # Simple implementation - doesn't actually modify the board
        return board, -player
        
    def _action_to_coords(self, action):
        row = action // self.size
        col = action % self.size
        return row, col
        
    def _coords_to_action(self, row, col):
        return row * self.size + col

class TestNode(unittest.TestCase):
    """Test the Node class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = SimpleGame()
        self.board = SimpleBoard()
        self.policy = np.ones(9) / 9  # Uniform policy for 3x3 board
        
    def test_init(self):
        """Test node initialization."""
        node = Node(self.game)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(len(node.children), 0)
        
    def test_expand(self):
        """Test node expansion."""
        node = Node(self.game)
        node.expand(self.board, 1, self.policy)
        
        # Check that children were created for all valid moves
        self.assertEqual(len(node.children), 9)
        
        # Check child properties
        for action, child in node.children.items():
            self.assertEqual(child.parent, node)
            self.assertEqual(child.action, action)
            self.assertAlmostEqual(child.prior, 1.0/9.0)
            
    def test_expand_terminal(self):
        """Test expansion of terminal node."""
        # Create a game that always returns terminal
        terminal_game = SimpleGame(game_result=1)
        node = Node(terminal_game)
        node.expand(self.board, 1, self.policy)
        
        # Node should be marked as terminal
        self.assertTrue(node.is_terminal)
        self.assertEqual(node.terminal_value, 1)
        self.assertEqual(len(node.children), 0)
        
    def test_select_child(self):
        """Test child selection using UCB."""
        # Create a game with only one valid move (action 1)
        valid_moves = np.zeros(9)
        valid_moves[1] = 1  # Only action 1 is valid
        limited_game = SimpleGame(valid_moves=valid_moves)
        
        # Create node with this limited game
        node = Node(limited_game)
        
        # Use uniform policy
        policy = np.ones(9) / 9
        
        # Expand the node
        node.expand(self.board, 1, policy)
        
        # Select a child - should only be able to select the valid move
        action, child = node.select_child()
        
        # Since only action 1 is valid, it must be selected
        self.assertEqual(action, 1)
        
    def test_update(self):
        """Test node update."""
        node = Node(self.game)
        node.update(0.5)
        self.assertEqual(node.visits, 1)
        self.assertEqual(node.value_sum, 0.5)
        
        node.update(-0.3)
        self.assertEqual(node.visits, 2)
        self.assertEqual(node.value_sum, 0.2)
        
    def test_get_value(self):
        """Test getting node value."""
        node = Node(self.game)
        self.assertEqual(node.get_value(), 0.0)  # Default value
        
        node.visits = 5
        node.value_sum = 3.0
        self.assertEqual(node.get_value(), 0.6)
        
    def test_get_children_distribution(self):
        """Test getting children distribution."""
        node = Node(self.game)
        node.expand(self.board, 1, self.policy)
        
        # Set different visit counts
        node.children[0].visits = 10
        node.children[1].visits = 5
        node.children[2].visits = 3
        node.children[3].visits = 1
        
        # Test with temperature 1.0 (proportional to visit counts)
        probs = node.get_children_distribution(temperature=1.0)
        expected_sum = 19  # Total visits
        self.assertAlmostEqual(probs[0], 10/expected_sum)
        self.assertAlmostEqual(probs[1], 5/expected_sum)
        self.assertAlmostEqual(probs[2], 3/expected_sum)
        self.assertAlmostEqual(probs[3], 1/expected_sum)
        
        # Test with temperature 0 (deterministic)
        probs = node.get_children_distribution(temperature=0)
        self.assertEqual(probs[0], 1.0)  # Most visited
        self.assertEqual(probs[1], 0.0)
        
        # Test with temperature 0.5 (higher temperatures give more uniform distribution)
        probs = node.get_children_distribution(temperature=0.5)
        self.assertGreater(probs[0], probs[1])  # Still in order of visits
        # But ratios should be different - can't easily test exact values

class TestMCTS(unittest.TestCase):
    """Test the MCTS class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = SimpleGame()
        self.board = SimpleBoard()
        
        # Create mock neural network
        self.policy = np.ones(9) / 9  # Uniform policy
        self.value = 0.0  # Neutral value
        self.neural_net = MockNeuralNetwork(self.policy, self.value)
        
        # Create MCTS with small number of simulations for testing
        self.mcts = MCTS(
            game=self.game,
            neural_net=self.neural_net,
            num_simulations=10,
            verbose=0  # Disable logging
        )
        
    def test_init(self):
        """Test MCTS initialization."""
        self.assertEqual(self.mcts.num_simulations, 10)
        self.assertEqual(self.mcts.cpuct, 1.0)
        self.assertEqual(self.mcts.temperature, 1.0)
        
    def test_search(self):
        """Test MCTS search."""
        action_probs, root = self.mcts.search(self.board, 1)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(np.sum(action_probs), 1.0)
        
        # Check that root node has been expanded
        self.assertTrue(root.is_expanded())
        
        # Check that the total visits match num_simulations
        self.assertEqual(root.visits, self.mcts.num_simulations)
        
    def test_search_with_exploration_noise(self):
        """Test MCTS search with Dirichlet noise."""
        # Run search with exploration noise
        action_probs, root = self.mcts.search(self.board, 1, add_exploration_noise=True)
        
        # Still should have valid probabilities and correct visits
        self.assertAlmostEqual(np.sum(action_probs), 1.0)
        self.assertEqual(root.visits, self.mcts.num_simulations)
        
    def test_simulate(self):
        """Test a single MCTS simulation."""
        # Create root node
        root = Node(self.game)
        root.expand(self.board, 1, self.policy)
        
        # Run a simulation
        value = self.mcts._simulate(root)
        
        # Check that the simulation returned a value
        self.assertIsInstance(value, float)
        
        # Check that the root node was updated
        self.assertEqual(root.visits, 1)
        
    def test_simulate_terminal(self):
        """Test simulation from terminal node."""
        # Create a game that always returns terminal
        terminal_game = SimpleGame(game_result=1)
        self.mcts.game = terminal_game
        
        # Create root node
        root = Node(terminal_game)
        root.expand(self.board, 1, self.policy)
        
        # Run a simulation
        value = self.mcts._simulate(root)
        
        # Check that the terminal value was returned
        self.assertEqual(value, 1)
        
        # Check that the root node was updated
        self.assertEqual(root.visits, 1)
        
    def test_parallel_simulations(self):
        """Test parallel MCTS simulations."""
        # Create MCTS with multiple threads
        parallel_mcts = MCTS(
            game=self.game,
            neural_net=self.neural_net,
            num_simulations=10,
            num_threads=2,
            verbose=0
        )
        
        # Create root node
        root = Node(self.game)
        root.expand(self.board, 1, self.policy)
        
        # Run parallel simulations
        parallel_mcts._run_parallel_simulations(root)
        
        # Check that the root node was updated correctly
        self.assertEqual(root.visits, 10)
        
    def test_select_action(self):
        """Test action selection."""
        # Test with temperature 0 (deterministic)
        action = self.mcts.select_action(self.board, 1, temperature=0)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.game.getActionSize())
        
    def test_reuse_tree(self):
        """Test tree reuse."""
        # Run search to build a tree
        _, root = self.mcts.search(self.board, 1)
        
        # Setup a child node with high visit count
        child_action = 0
        child_node = root.children[child_action]
        child_node.visits = 100  # High visit count
        
        # Reuse tree for the same action
        new_root = self.mcts.reuse_tree(root, self.board, -1, child_action)
        
        # Check that the child node was reused
        self.assertEqual(new_root.visits, 100)
        self.assertIsNone(new_root.parent)  # Parent should be set to None
        
        # Try reusing for an action not in the tree
        non_existent_action = 9999
        new_root = self.mcts.reuse_tree(root, self.board, -1, non_existent_action)
        
        # Should get a fresh node
        self.assertEqual(new_root.visits, 0)
        
    def test_tree_structure(self):
        """Test that the tree structure is built correctly."""
        # Run search with few simulations
        _, root = self.mcts.search(self.board, 1)
        
        # Check that the root has children
        self.assertTrue(len(root.children) > 0)
        
        # Check that the children have valid properties
        for action, child in root.children.items():
            self.assertEqual(child.parent, root)
            self.assertEqual(child.action, action)
            self.assertIsInstance(child.visits, int)
            
    def test_visualize_tree(self):
        """Test tree visualization."""
        # Run search with few simulations
        _, root = self.mcts.search(self.board, 1)
        
        # Test visualization without file output
        tree_str = self.mcts.visualize_tree(root, max_depth=2)
        
        # Check that the string contains node information
        self.assertIn("Node: visits=", tree_str)
        
        # Test with file output
        file_path = "test_tree.txt"
        self.mcts.visualize_tree(root, max_depth=2, file_path=file_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(file_path))
        
        # Cleanup
        os.remove(file_path)
        
    def test_ucb_values(self):
        """Test UCB formula with predetermined values."""
        # Create a node with specific values
        node = Node(self.game)
        node.expand(self.board, 1, self.policy)
        
        # Set specific values for children
        child0 = node.children[0]
        child0.visits = 10
        child0.value_sum = 5  # value = 0.5
        child0.prior = 0.2
        
        child1 = node.children[1]
        child1.visits = 5
        child1.value_sum = 4  # value = 0.8
        child1.prior = 0.4
        
        # Calculate UCB values manually
        sum_visits = 15
        c_puct = 1.0
        
        ucb0 = 0.5 + c_puct * 0.2 * math.sqrt(sum_visits) / (1 + 10)
        ucb1 = 0.8 + c_puct * 0.4 * math.sqrt(sum_visits) / (1 + 5)
        
        # Child 1 should have higher UCB
        self.assertGreater(ucb1, ucb0)
        
        # Test select_child
        action, child = node.select_child(c_puct=c_puct)
        self.assertEqual(action, 1)  # Should select child1 with higher UCB
        
    def test_backpropagation(self):
        """Test backpropagation through multiple levels."""
        # Create a simple tree structure
        root = Node(self.game)
        root.expand(self.board, 1, self.policy)
        
        child = root.children[0]
        child.expand(self.board, -1, self.policy)  # Opposite player
        
        grandchild = child.children[0]
        
        # Perform backpropagation manually
        value = 0.8
        
        # Update from grandchild to root
        grandchild.update(value)
        child.update(-value)  # Negate for opposite player
        root.update(value)  # Back to original player
        
        # Check values
        self.assertEqual(grandchild.visits, 1)
        self.assertEqual(grandchild.value_sum, value)
        
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.value_sum, -value)
        
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.value_sum, value)
        
    def test_different_temperature(self):
        """Test different temperature settings."""
        # Create a node with different visit counts
        node = Node(self.game)
        node.expand(self.board, 1, self.policy)
        
        # Set visits to different values
        node.children[0].visits = 100
        node.children[1].visits = 50
        node.children[2].visits = 10
        
        # Test with temperature 1.0
        dist1 = node.get_children_distribution(temperature=1.0)
        
        # Test with temperature 0.5
        dist2 = node.get_children_distribution(temperature=0.5)
        
        # Test with temperature 0.1
        dist3 = node.get_children_distribution(temperature=0.1)
        
        # Test with temperature 0 (deterministic)
        dist4 = node.get_children_distribution(temperature=0)
        
        # Verify that lower temperatures make the distribution more peaked
        # at the most visited action (0)
        self.assertLess(dist1[0] - dist1[1], dist2[0] - dist2[1])
        self.assertLess(dist2[0] - dist2[1], dist3[0] - dist3[1])
        self.assertEqual(dist4[0], 1.0)  # Deterministic
        
    def test_integration_with_neural_network(self):
        """Test integration with neural network."""
        # Create a real game and neural network
        try:
            game = YinYangGame(n=3, m=3)  # Small board for testing
            neural_net = YinYangNeuralNetwork(game, num_channels=16, num_res_blocks=1)
            
            # Create MCTS with the real components
            mcts = MCTS(
                game=game,
                neural_net=neural_net,
                num_simulations=5,  # Very few simulations for testing
                verbose=0
            )
            
            # Create initial board
            board = game.getInitBoard()
            
            # Run search
            action_probs, root = mcts.search(board, 1)
            
            # Check that we get valid probabilities
            self.assertAlmostEqual(np.sum(action_probs), 1.0)
            
            # Check that the neural network was called
            self.assertTrue(len(root.children) > 0)
        except ImportError:
            # Skip test if neural network module is not available
            self.skipTest("Neural network module not available")
            
    def test_mcts_for_simple_position(self):
        """Test MCTS for a simple position with known best move."""
        # Create a game with limited valid moves
        valid_moves = np.zeros(9)
        valid_moves[0] = 1  # Only move 0 is valid
        limited_game = SimpleGame(valid_moves=valid_moves)
        
        mcts = MCTS(
            game=limited_game,
            neural_net=self.neural_net,
            num_simulations=10,
            verbose=0
        )
        
        # Run search
        action_probs, _ = mcts.search(self.board, 1)
        
        # Check that only move 0 has non-zero probability
        self.assertEqual(np.argmax(action_probs), 0)
        self.assertAlmostEqual(action_probs[0], 1.0)
        
if __name__ == '__main__':
    unittest.main() 