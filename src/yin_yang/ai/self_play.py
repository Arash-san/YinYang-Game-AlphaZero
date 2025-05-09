import os
import numpy as np
import time
import logging
import random
import threading
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from .mcts import MCTS
from .neural_network import YinYangNeuralNetwork

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YinYangSelfPlay')

class SelfPlayWorker:
    """
    Worker class for generating self-play games.
    """
    def __init__(self, game, model_path, num_simulations=800, num_games=1, 
                 temperature_threshold=10, dirichlet_alpha=0.3, 
                 dirichlet_epsilon=0.25, cpuct=1.0, num_parallel=1):
        """
        Initialize a self-play worker.
        
        Args:
            game: Game instance
            model_path: Path to the neural network model
            num_simulations: Number of MCTS simulations per move
            num_games: Number of games to play
            temperature_threshold: After this many moves, temperature is set to 0
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Epsilon parameter for Dirichlet noise
            cpuct: Exploration constant for MCTS
            num_parallel: Number of parallel threads for MCTS
        """
        self.game = game
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.cpuct = cpuct
        self.num_parallel = num_parallel
        
        # Create neural network
        self.neural_net = YinYangNeuralNetwork(game)
        if os.path.exists(model_path):
            self.neural_net.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"No model found at {model_path}, using randomly initialized model")
        
        # Create MCTS
        self.mcts = MCTS(
            game=game,
            neural_net=self.neural_net,
            num_simulations=num_simulations,
            cpuct=cpuct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            num_threads=num_parallel
        )
        
    def play_game(self) -> List[Tuple]:
        """
        Play a single game of self-play.
        
        Returns:
            examples: List of (board, policy, value) tuples
        """
        examples = []
        board = self.game.getInitBoard()
        player = 1  # Player 1 (Black) starts
        step = 0
        
        # Root node for tree reuse
        root = None
        
        # Keep track of consecutive passes
        consecutive_passes = 0
        max_consecutive_passes = 2
        
        while True:
            # Set temperature based on move number
            temperature = 1.0 if step < self.temperature_threshold else 0
            
            # Get the canonical form of the board (from current player's perspective)
            canonical_board = self.game.getCanonicalForm(board, player)
            
            # Check for valid moves
            valid_moves = self.game.getValidMoves(canonical_board, 1)
            valid_moves_indices = np.where(valid_moves == 1)[0]
            
            # Check if there are any valid moves
            if len(valid_moves_indices) == 0:
                # No valid moves for current player
                consecutive_passes += 1
                logger.info(f"Player {player} has no valid moves (pass). Consecutive passes: {consecutive_passes}")
                
                if consecutive_passes >= max_consecutive_passes:
                    # Game over due to consecutive passes
                    game_result = self.game.getGameEnded(board, player)
                    if game_result == 0:  # If game hasn't ended yet, set it as a draw
                        game_result = 1e-4  # Small non-zero value to indicate a draw
                    
                    # Fill in values for all examples and return
                    value = game_result
                    for i in range(len(examples)):
                        examples[i] = (examples[i][0], examples[i][1], value if i % 2 == 0 else -value)
                        value = -value
                    
                    logger.info(f"Game finished (consecutive passes) after {step} steps. Result: {game_result}.")
                    return examples
                
                # Skip to next player
                player = -player
                continue
            
            # Reset consecutive passes counter since we have valid moves
            consecutive_passes = 0
            
            # Add exploration noise at the root node
            add_noise = (step == 0)
            
            # Get action probabilities from MCTS
            if root is None:
                pi, root = self.mcts.search(canonical_board, 1, add_exploration_noise=add_noise)
            else:
                pi, root = self.mcts.search(canonical_board, 1, add_exploration_noise=add_noise)
            
            # Store the training example before taking action
            examples.append((canonical_board, pi, None))  # Value will be filled in later
            
            # Apply temperature to policy
            if temperature == 0:
                # Deterministic choice
                best_moves = np.where(pi == np.max(pi))[0]
                action = np.random.choice(best_moves)
            else:
                # Sample action based on policy probabilities
                action_probs = pi.copy()
                # Zero out invalid moves
                action_probs = action_probs * valid_moves
                # Normalize
                if np.sum(action_probs) > 0:
                    action_probs = action_probs / np.sum(action_probs)
                else:
                    # Fallback to uniform distribution over valid moves
                    action_probs = np.zeros_like(valid_moves)
                    action_probs[valid_moves_indices] = 1.0 / len(valid_moves_indices)
                
                action = np.random.choice(len(action_probs), p=action_probs)
            
            # Make the move
            board, player = self.game.getNextState(board, player, action)
            step += 1
            
            # Check if game is over
            game_result = self.game.getGameEnded(board, player)
            if game_result != 0:
                # Game is over, fill in the values for all examples
                value = game_result
                
                # Fill in values for all examples based on final result and perspective
                for i in range(len(examples)):
                    # The value is from the perspective of the player who just made a move
                    # If i is even, it's from the first player's perspective
                    # If i is odd, it's from the second player's perspective
                    # So we need to flip the sign of the value for odd i
                    examples[i] = (examples[i][0], examples[i][1], value if i % 2 == 0 else -value)
                    
                    # Flip the value for the next example (from opponent's perspective)
                    value = -value
                
                # Log the game result
                board_size = self.game.getBoardSize()
                logger.info(f"Game finished after {step} steps. "
                           f"Result: {game_result}. Board size: {board_size}")
                
                return examples
            
            # Reuse tree for the next move to improve efficiency
            # Find the child node corresponding to the action we took
            root = self.mcts.reuse_tree(root, board, player, action)
    
    def generate_games(self) -> List[Tuple]:
        """
        Generate multiple self-play games.
        
        Returns:
            all_examples: List of (board, policy, value) tuples from all games
        """
        all_examples = []
        
        for i in tqdm(range(self.num_games), desc="Self-Play Games"):
            try:
                # Play a game
                examples = self.play_game()
                all_examples.extend(examples)
                
                # Log progress
                logger.info(f"Completed game {i+1}/{self.num_games} with {len(examples)} examples")
            except Exception as e:
                # Log the error but continue with the next game
                logger.error(f"Error in game {i+1}: {str(e)}")
                continue
        
        return all_examples

class SelfPlayManager:
    """
    Manager class for parallel self-play game generation.
    """
    def __init__(self, game, model_path, num_workers=1, num_simulations=800, 
                 games_per_worker=1, temperature_threshold=10, 
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, cpuct=1.0,
                 mcts_parallel=1):
        """
        Initialize a self-play manager.
        
        Args:
            game: Game instance
            model_path: Path to the neural network model
            num_workers: Number of parallel self-play workers
            num_simulations: Number of MCTS simulations per move
            games_per_worker: Number of games per worker
            temperature_threshold: After this many moves, temperature is set to 0
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Epsilon parameter for Dirichlet noise
            cpuct: Exploration constant for MCTS
            mcts_parallel: Number of parallel threads for MCTS in each worker
        """
        self.game = game
        self.model_path = model_path
        self.num_workers = num_workers
        self.num_simulations = num_simulations
        self.games_per_worker = games_per_worker
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.cpuct = cpuct
        self.mcts_parallel = mcts_parallel
        
        logger.info(f"SelfPlayManager initialized with {num_workers} workers, "
                   f"{games_per_worker} games per worker, {num_simulations} MCTS simulations")
        
    def _worker_task(self, return_queue, worker_id):
        """
        Self-play worker task.
        
        Args:
            return_queue: Queue to put the generated examples
            worker_id: ID of the worker
        """
        try:
            # Create a worker
            worker = SelfPlayWorker(
                game=self.game,
                model_path=self.model_path,
                num_simulations=self.num_simulations,
                num_games=self.games_per_worker,
                temperature_threshold=self.temperature_threshold,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                cpuct=self.cpuct,
                num_parallel=self.mcts_parallel
            )
            
            # Generate games
            examples = worker.generate_games()
            
            # Put examples in the queue
            return_queue.put(examples)
            logger.info(f"Worker {worker_id} completed {self.games_per_worker} games")
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered an error: {str(e)}")
            # Put an empty list in the queue so the main process doesn't hang
            return_queue.put([])
        
    def generate_games_parallel(self) -> List[Tuple]:
        """
        Generate multiple self-play games in parallel.
        
        Returns:
            all_examples: List of (board, policy, value) tuples from all games
        """
        # Create a queue for returning examples
        return_queue = mp.Queue()
        
        # Start processes
        processes = []
        for i in range(self.num_workers):
            p = mp.Process(target=self._worker_task, args=(return_queue, i))
            p.daemon = True  # Make sure processes terminate when main process exits
            p.start()
            processes.append(p)
        
        # Collect results
        all_examples = []
        active_workers = len(processes)
        
        # Wait for all workers to finish or handle their failure
        while active_workers > 0:
            try:
                examples = return_queue.get(timeout=60)  # Wait for 60 seconds for results
                all_examples.extend(examples)
                active_workers -= 1
            except Exception as e:
                # Check if any process has terminated unexpectedly
                for i, p in enumerate(processes):
                    if not p.is_alive() and p.exitcode != 0:
                        logger.error(f"Worker {i} failed with exit code {p.exitcode}")
                        active_workers -= 1
                
                if active_workers == 0:
                    break
        
        # Wait for all processes to finish (just to be sure)
        for p in processes:
            p.join(timeout=1)
            if p.is_alive():
                logger.warning("Terminating worker process that didn't finish")
                p.terminate()
        
        logger.info(f"Generated {len(all_examples)} examples")
        
        return all_examples

def generate_self_play_data(game, model_path, output_dir, num_games=100, 
                           num_workers=1, num_simulations=800):
    """
    Generate self-play data and save it to file.
    
    Args:
        game: Game instance
        model_path: Path to the neural network model
        output_dir: Directory to save the generated data
        num_games: Total number of games to generate
        num_workers: Number of parallel workers
        num_simulations: Number of MCTS simulations per move
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate games per worker
    games_per_worker = max(1, num_games // num_workers)
    
    # Create self-play manager
    manager = SelfPlayManager(
        game=game,
        model_path=model_path,
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        num_simulations=num_simulations
    )
    
    # Generate games
    examples = manager.generate_games_parallel()
    
    # Save examples to file
    timestamp = int(time.time())
    filename = os.path.join(output_dir, f"self_play_data_{timestamp}.npz")
    
    # Extract the components from the examples
    boards = [ex[0] for ex in examples]
    policies = [ex[1] for ex in examples]
    values = [ex[2] for ex in examples]
    
    # Save to file
    np.savez(
        filename,
        boards=boards,
        policies=policies,
        values=values
    )
    
    logger.info(f"Saved {len(examples)} examples to {filename}")
    
    return filename 