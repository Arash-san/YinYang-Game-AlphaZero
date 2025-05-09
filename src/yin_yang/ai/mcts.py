import numpy as np
import math
import time
import logging
import os
import threading
import sys
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Union

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    filename='mcts.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YinYangMCTS')

# Constants for the UCB formula
CPUCT = 1.0  # Exploration constant

class Node:
    """
    Node class for the MCTS tree.
    """
    def __init__(self, game, parent=None, action=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: Dict[int, Node] = {}  # Map from action to child node
        
        # Node statistics
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior  # Prior probability from policy network
        
        # Game state information
        self.board = None
        self.player = None
        self.valid_moves = None
        self.is_terminal = False
        self.terminal_value = None
        
    def expand(self, board, player, policy_probs):
        """
        Expand the node with child nodes for all valid moves.
        
        Args:
            board: Current board state
            player: Current player (1 for Black, -1 for White)
            policy_probs: Policy probabilities from neural network
        """
        self.board = board
        self.player = player
        
        # Check if the game is over
        game_result = self.game.getGameEnded(board, player)
        if game_result != 0:
            self.is_terminal = True
            self.terminal_value = game_result
            logger.debug(f"Node is terminal with value {self.terminal_value}")
            return
        
        # Get valid moves
        self.valid_moves = self.game.getValidMoves(board, player)
        valid_moves_indices = np.where(self.valid_moves == 1)[0]
        
        # Create child nodes for each valid move
        for action in valid_moves_indices:
            # Mask the policy probabilities with valid moves and normalize
            if policy_probs is not None:
                prior = policy_probs[action]
            else:
                # If no policy probs provided, use uniform distribution
                prior = 1.0 / len(valid_moves_indices)
                
            # Create the child node
            self.children[action] = Node(
                game=self.game,
                parent=self,
                action=action,
                prior=prior
            )
            
        logger.debug(f"Expanded node for player {player} with {len(self.children)} children")
            
    def is_expanded(self):
        """Check if the node has been expanded."""
        return len(self.children) > 0 or self.is_terminal
    
    def select_child(self, c_puct=CPUCT):
        """
        Select the child node using the UCB formula.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            action, child: Selected action and child node
        """
        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        # where Q(s,a) is the mean value, P(s,a) is the prior from policy network,
        # N(s,a) is the visit count.
        
        # Count sum of all child visits
        sum_visits = sum(child.visits for child in self.children.values())
        
        # Find child with highest UCB value
        best_ucb = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # Skip invalid moves
            if self.valid_moves is not None and self.valid_moves[action] == 0:
                continue
                
            # Calculate exploitation term (Q-value)
            q_value = 0.0
            if child.visits > 0:
                q_value = child.value_sum / child.visits
                
            # Using AlphaZero UCB formula
            ucb = q_value + c_puct * child.prior * math.sqrt(sum_visits) / (1 + child.visits)
            
            # Update best child if needed
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child
                
        if best_action is None:
            # No valid move found (shouldn't happen if expanded properly)
            logger.error("No valid move found in select_child")
            # Return a random child as fallback
            best_action, best_child = random.choice(list(self.children.items()))
        
        logger.debug(f"Selected child with action {best_action}, UCB value {best_ucb}")
        return best_action, best_child
    
    def update(self, value):
        """
        Update node statistics with simulation result.
        
        Args:
            value: Result of the simulation (-1, 0, or 1)
        """
        self.visits += 1
        self.value_sum += value
        logger.debug(f"Updated node: visits={self.visits}, value_sum={self.value_sum}")
        
    def get_value(self):
        """Get the mean value of the node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_visit_count(self):
        """Get the visit count of the node."""
        return self.visits
    
    def get_children_visit_counts(self):
        """
        Get the visit counts of all children.
        
        Returns:
            counts: numpy array of visit counts for all possible actions
        """
        action_size = self.game.getActionSize()
        counts = np.zeros(action_size)
        
        for action, child in self.children.items():
            counts[action] = child.visits
            
        return counts
    
    def get_children_distribution(self, temperature=1.0):
        """
        Get the probability distribution over children based on visit counts.
        
        Args:
            temperature: Temperature parameter for exploration/exploitation
        
        Returns:
            probs: numpy array of probabilities for all possible actions
        """
        action_size = self.game.getActionSize()
        counts = np.zeros(action_size)
        
        for action, child in self.children.items():
            counts[action] = child.visits
            
        # Apply temperature - lower means more exploitation, higher means more exploration
        if temperature == 0:  # Deterministic choice
            best_actions = np.where(counts == np.max(counts))[0]
            probs = np.zeros(action_size)
            probs[best_actions] = 1.0 / len(best_actions)
        else:
            # Apply temperature and normalize
            if temperature != 1.0:
                counts = np.power(counts, 1.0 / temperature)
            
            if np.sum(counts) > 0:
                probs = counts / np.sum(counts)
            else:
                # Fallback to uniform distribution
                probs = np.ones(action_size) / action_size
                
        return probs
    
    def __str__(self):
        """String representation of the node for debugging."""
        if self.is_terminal:
            return f"Terminal Node (value={self.terminal_value})"
        
        return (f"Node(visits={self.visits}, "
                f"value={self.get_value():.3f}, "
                f"player={self.player}, "
                f"children={len(self.children)})")

class MCTS:
    """
    Monte Carlo Tree Search algorithm with neural network integration.
    """
    def __init__(self, game, neural_net, num_simulations=800, cpuct=1.0, 
                 temperature=1.0, num_threads=1, dirichlet_noise=True,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, verbose=1):
        """
        Initialize the MCTS algorithm.
        
        Args:
            game: Game object that implements the required methods
            neural_net: Neural network object that implements predict method
            num_simulations: Number of simulations per search
            cpuct: Exploration constant in UCB formula
            temperature: Temperature parameter for move selection
            num_threads: Number of threads for parallel MCTS
            dirichlet_noise: Whether to add Dirichlet noise to root node priors
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            dirichlet_epsilon: Epsilon parameter for Dirichlet noise
            verbose: Verbosity level for logging (0=ERROR, 1=INFO, 2=DEBUG)
        """
        self.game = game
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.temperature = temperature
        self.num_threads = max(1, num_threads)  # Ensure at least 1 thread
        
        # Dirichlet noise parameters for exploration
        self.use_dirichlet = dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Thread lock for parallel MCTS
        self.lock = threading.Lock()
        
        # Set logging level
        if verbose == 0:
            logger.setLevel(logging.ERROR)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"MCTS initialized with {num_simulations} simulations, "
                   f"cpuct={cpuct}, temperature={temperature}, threads={num_threads}")
        
    def search(self, board, player, add_exploration_noise=False):
        """
        Perform MCTS search from the given board state.
        
        Args:
            board: Current board state
            player: Current player (1 for Black, -1 for White)
            add_exploration_noise: Whether to add Dirichlet noise to root node priors
            
        Returns:
            action_probs: Action probabilities based on visit counts
            root_node: Root node of the search tree for reuse
        """
        # Create root node
        root = Node(self.game)
        # Initialize the root node with the current board and player
        root.board = board
        root.player = player
        
        # Get policy and value prediction from neural network
        policy, value = self.neural_net.predict(board)
        
        # Add exploration noise to root node priors if specified
        if add_exploration_noise and self.use_dirichlet:
            # Add Dirichlet noise to root node for exploration
            valid_moves = self.game.getValidMoves(board, player)
            valid_indices = np.where(valid_moves == 1)[0]
            
            if len(valid_indices) > 0:  # Only add noise if there are valid moves
                # Generate Dirichlet noise for valid moves
                noise = np.random.dirichlet(
                    [self.dirichlet_alpha] * len(valid_indices)
                )
                
                # Apply noise to policy
                for i, action in enumerate(valid_indices):
                    policy[action] = (1 - self.dirichlet_epsilon) * policy[action] + \
                                    self.dirichlet_epsilon * noise[i]
                
                logger.info("Added Dirichlet noise to root node priors")
            
        # Expand root node
        root.expand(board, player, policy)
        
        # Run simulations
        if self.num_threads > 1:
            # Parallel simulations
            self._run_parallel_simulations(root)
        else:
            # Sequential simulations
            for _ in range(self.num_simulations):
                self._simulate(root)
        
        # Calculate move probabilities based on visit counts
        action_probs = root.get_children_distribution(self.temperature)
        
        # Log search results
        visits = root.get_children_visit_counts()
        top_actions = np.argsort(visits)[-5:][::-1]  # Top 5 most visited actions
        
        visit_info = ", ".join([
            f"{action}({self.game._action_to_coords(action)}): {visits[action]}" 
            for action in top_actions if visits[action] > 0
        ])
        
        logger.info(f"MCTS search completed with {root.visits} total visits. "
                   f"Top actions: {visit_info}")
        
        return action_probs, root
    
    def _simulate(self, node):
        """
        Run a single MCTS simulation from the given node.
        
        Args:
            node: Starting node for simulation
        
        Returns:
            value: Simulation result
        """
        # Phase 1: Selection - Travel down the tree until we reach a leaf node
        current = node
        path = [current]
        
        # Select until we reach a leaf node
        while current.is_expanded() and not current.is_terminal:
            action, current = current.select_child(self.cpuct)
            path.append(current)
        
        # If we reached a terminal node, use its value
        if current.is_terminal:
            value = current.terminal_value
            logger.debug(f"Simulation reached terminal node with value {value}")
        else:
            # Phase 2: Expansion - Expand the leaf node if not terminal
            # Handle the case where path is too short (root node is not expanded yet)
            if len(path) < 2:
                # We're at the root node and it's not expanded yet
                # Use the root node's board and player
                board = node.board
                player = node.player
                
                # Get neural network prediction for the root node
                policy, value = self.neural_net.predict(board)
                
                # Expand the root node
                node.expand(board, player, policy)
                logger.debug(f"Expanded root node with value {value}")
            else:
                # Normal case - get board and player from parent node
                board = path[-2].board
                player = path[-2].player
                
                if not current.is_expanded():
                    # Get the new board state after taking the action
                    action = current.action
                    new_board, next_player = self.game.getNextState(board, player, action)
                    
                    # Get policy and value prediction from neural network
                    policy, value = self.neural_net.predict(new_board)
                    
                    # Expand the node
                    current.expand(new_board, next_player, policy)
                    
                    logger.debug(f"Expanded node at depth {len(path)} with value {value}")
                else:
                    # This should not happen if the loop condition is correct
                    logger.error("Unexpected: Node is already expanded but not marked as such")
                    value = 0.0
        
        # Phase 4: Backpropagation - Update the statistics of all nodes in the path
        for node in reversed(path):
            # The value is from the perspective of the player at the node
            # If the current player is different from the original player, negate the value
            if node != path[-1] and node.player != path[-1].player:
                node.update(-value)
            else:
                node.update(value)
        
        return value
    
    def _run_parallel_simulations(self, root):
        """Run multiple MCTS simulations in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit simulation tasks
            futures = [executor.submit(self._simulate, root) 
                      for _ in range(self.num_simulations)]
            
            # Wait for all simulations to complete
            for future in futures:
                future.result()
                
    def select_action(self, board, player, temperature=None, valid_moves=None, add_exploration_noise=False):
        """
        Select an action based on MCTS search.
        
        Args:
            board: Current board state
            player: Current player (1 for Black, -1 for White)
            temperature: Temperature parameter (overwrites instance value if provided)
            valid_moves: Binary vector of valid moves (1=valid, 0=invalid)
            add_exploration_noise: Whether to add Dirichlet noise
            
        Returns:
            action: Selected action
        """
        # Use provided temperature or instance default
        temp = temperature if temperature is not None else self.temperature
        
        # Run MCTS search
        action_probs, _ = self.search(board, player, add_exploration_noise)
        
        # If valid_moves provided, filter out invalid moves
        if valid_moves is not None:
            # Ensure valid_moves and action_probs have the same length
            if len(valid_moves) != len(action_probs):
                logger.error(f"Valid moves length {len(valid_moves)} != action probs length {len(action_probs)}")
                valid_moves = np.ones_like(action_probs)
            
            # Mask out invalid moves
            masked_probs = action_probs * valid_moves
            sum_probs = np.sum(masked_probs)
            
            # If no valid moves left after masking, return most likely move from original distribution
            if sum_probs <= 0:
                logger.warning("No valid moves with non-zero probability")
                # Fallback to selecting the most likely move from original distribution
                return np.argmax(action_probs)
            
            # Normalize probabilities
            action_probs = masked_probs / sum_probs
        
        # Sample action based on the probabilities
        actions = np.arange(len(action_probs))
        
        if temp == 0:  # Deterministic choice
            action = np.argmax(action_probs)
        else:  # Sample from distribution
            action = np.random.choice(actions, p=action_probs)
            
        # Log the selected action
        x, y = self.game._action_to_coords(action)
        logger.info(f"Selected action {action} ({x},{y}) with probability {action_probs[action]:.4f}")
        
        return action
    
    def reuse_tree(self, old_root, board, player, action_taken):
        """
        Reuse the search tree for the next move.
        
        Args:
            old_root: Previous root node
            board: Current board state
            player: Current player (1 for Black, -1 for White)
            action_taken: Action taken to reach current state
            
        Returns:
            new_root: New root node reusing subtree if possible
        """
        # Check if the action is in the children of the old root
        if action_taken in old_root.children:
            # Reuse the subtree
            new_root = old_root.children[action_taken]
            new_root.parent = None  # Detach from parent
            
            logger.info(f"Reusing subtree with {new_root.visits} visits")
            return new_root
        else:
            # Cannot reuse tree, return a fresh node
            logger.info("Cannot reuse tree, creating new root")
            return Node(self.game)
            
    def visualize_tree(self, root, max_depth=3, file_path=None):
        """
        Visualize the MCTS tree for debugging.
        
        Args:
            root: Root node of the tree
            max_depth: Maximum depth to visualize
            file_path: Path to save the visualization (if None, print to console)
        """
        def _build_tree_string(node, depth, prefix=""):
            if depth > max_depth:
                return [f"{prefix}... (max depth reached)"]
                
            result = []
            
            # Node information
            node_info = f"{prefix}Node: visits={node.visits}, value={node.get_value():.3f}"
            if node.action is not None:
                x, y = self.game._action_to_coords(node.action)
                node_info += f", action=({x},{y})"
            result.append(node_info)
            
            # Children information
            if node.is_expanded():
                # Sort children by visit count
                sorted_children = sorted(
                    node.children.items(), 
                    key=lambda x: x[1].visits, 
                    reverse=True
                )
                
                # Add top 3 children
                for i, (action, child) in enumerate(sorted_children[:3]):
                    if i < len(sorted_children) - 1:
                        result.extend(_build_tree_string(
                            child, depth + 1, prefix + "├── "
                        ))
                    else:
                        result.extend(_build_tree_string(
                            child, depth + 1, prefix + "└── "
                        ))
                        
                # Indicate if there are more children
                if len(sorted_children) > 3:
                    result.append(f"{prefix}└── ... ({len(sorted_children) - 3} more children)")
            
            return result
        
        # Build tree visualization
        tree_lines = _build_tree_string(root, 0)
        tree_str = "\n".join(tree_lines)
        
        # Output the visualization
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(tree_str)
        else:
            print(tree_str)
            
        logger.info(f"Tree visualization created with max depth {max_depth}")
        
        return tree_str 