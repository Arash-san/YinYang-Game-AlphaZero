import os
import numpy as np
import time
import logging
import argparse
from typing import List, Dict, Tuple, Optional

from .self_play import SelfPlayManager, generate_self_play_data
from .training_pipeline import TrainingPipeline, run_training_pipeline
from .neural_network import YinYangNeuralNetwork
from .mcts import MCTS

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YinYangAlphaZero')

class AlphaZero:
    """
    Main AlphaZero implementation for the Yin-Yang game.
    """
    def __init__(self, game, model_dir='models', data_dir='data',
                 num_iterations=100, num_episodes=100, num_simulations=800,
                 num_epochs=10, temperature_threshold=10, update_threshold=0.6,
                 num_workers=1, mcts_threads=1):
        """
        Initialize the AlphaZero system.
        
        Args:
            game: Game instance
            model_dir: Directory to save/load models
            data_dir: Directory to save/load training data
            num_iterations: Number of training iterations to run
            num_episodes: Number of self-play episodes per iteration
            num_simulations: Number of MCTS simulations per move
            num_epochs: Number of training epochs per iteration
            temperature_threshold: After this many moves, temperature is set to 0
            update_threshold: Win ratio threshold to update the best model
            num_workers: Number of parallel self-play workers
            mcts_threads: Number of parallel threads for MCTS in each worker
        """
        self.game = game
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.num_epochs = num_epochs
        self.temperature_threshold = temperature_threshold
        self.update_threshold = update_threshold
        self.num_workers = num_workers
        self.mcts_threads = mcts_threads
        
        # Create directories if they don't exist
        for directory in [model_dir, data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Paths for the current and best models
        self.current_model_path = os.path.join(model_dir, "current_model.pth.tar")
        self.best_model_path = os.path.join(model_dir, "best_model.pth.tar")
        
        # Create initial model if it doesn't exist
        if not os.path.exists(self.current_model_path):
            self._initialize_model()
        
        # Copy current model to best model if it doesn't exist
        if not os.path.exists(self.best_model_path):
            import shutil
            shutil.copy(self.current_model_path, self.best_model_path)
        
        logger.info(f"AlphaZero initialized with model_dir={model_dir}, data_dir={data_dir}")
        logger.info(f"Parameters: iterations={num_iterations}, episodes={num_episodes}, "
                   f"simulations={num_simulations}, epochs={num_epochs}")
        
    def _initialize_model(self):
        """Initialize a new model and save it."""
        model = YinYangNeuralNetwork(self.game)
        model.save_model(self.current_model_path)
        logger.info(f"Initialized new model and saved to {self.current_model_path}")
        
    def self_play(self, model_path):
        """
        Run self-play to generate training data.
        
        Args:
            model_path: Path to the model to use for self-play
            
        Returns:
            data_file: Path to the generated data file
        """
        logger.info(f"Starting self-play with model {model_path}")
        
        # Generate self-play data
        data_file = generate_self_play_data(
            game=self.game,
            model_path=model_path,
            output_dir=self.data_dir,
            num_games=self.num_episodes,
            num_workers=self.num_workers,
            num_simulations=self.num_simulations
        )
        
        logger.info(f"Self-play completed. Data saved to {data_file}")
        return data_file
        
    def train(self):
        """
        Train the model on the generated data.
        
        Returns:
            new_model_path: Path to the trained model
        """
        logger.info("Starting model training")
        
        # Run training pipeline
        new_model_path = run_training_pipeline(
            game=self.game,
            model_dir=self.model_dir,
            data_dir=self.data_dir,
            num_iterations=1,
            sample_size=10000,
            checkpoint_interval=1
        )
        
        # Copy the trained model to the current model path
        import shutil
        shutil.copy(new_model_path, self.current_model_path)
        
        logger.info(f"Training completed. Model saved to {self.current_model_path}")
        return self.current_model_path
        
    def evaluate(self, current_model_path, best_model_path, num_games=40):
        """
        Evaluate the current model against the best model.
        
        Args:
            current_model_path: Path to the current model
            best_model_path: Path to the best model
            num_games: Number of games to play
            
        Returns:
            win_ratio: Ratio of games won by the current model
        """
        logger.info(f"Evaluating {current_model_path} against {best_model_path}")
        
        # Load the models
        current_model = YinYangNeuralNetwork(self.game)
        current_model.load_model(current_model_path)
        
        best_model = YinYangNeuralNetwork(self.game)
        best_model.load_model(best_model_path)
        
        # Create MCTS for both models
        current_mcts = MCTS(
            game=self.game,
            neural_net=current_model,
            num_simulations=self.num_simulations,
            num_threads=self.mcts_threads
        )
        
        best_mcts = MCTS(
            game=self.game,
            neural_net=best_model,
            num_simulations=self.num_simulations,
            num_threads=self.mcts_threads
        )
        
        # Play games
        current_wins = 0
        best_wins = 0
        draws = 0
        
        for i in range(num_games):
            # Alternate starting player
            if i % 2 == 0:
                first_player = current_mcts
                second_player = best_mcts
                first_is_current = True
            else:
                first_player = best_mcts
                second_player = current_mcts
                first_is_current = False
            
            # Play the game
            board = self.game.getInitBoard()
            player = 1  # Player 1 (Black) starts
            
            while True:
                # Get the player's move
                if player == 1:
                    action = first_player.select_action(board, player, temperature=0)
                else:
                    action = second_player.select_action(board, player, temperature=0)
                
                # Make the move
                board, player = self.game.getNextState(board, player, action)
                
                # Check if game is over
                game_result = self.game.getGameEnded(board, player)
                if game_result != 0:
                    # Game is over
                    if game_result == 1:  # First player won
                        if first_is_current:
                            current_wins += 1
                        else:
                            best_wins += 1
                    elif game_result == -1:  # Second player won
                        if first_is_current:
                            best_wins += 1
                        else:
                            current_wins += 1
                    else:  # Draw
                        draws += 1
                    break
        
        # Calculate win ratio
        win_ratio = current_wins / num_games
        
        logger.info(f"Evaluation completed. Current model wins: {current_wins}, "
                   f"Best model wins: {best_wins}, Draws: {draws}, Win ratio: {win_ratio:.2f}")
        
        return win_ratio
        
    def update_best_model(self, win_ratio):
        """
        Update the best model if the current model is better.
        
        Args:
            win_ratio: Ratio of games won by the current model
            
        Returns:
            updated: Whether the best model was updated
        """
        if win_ratio >= self.update_threshold:
            # Current model is better, update the best model
            import shutil
            shutil.copy(self.current_model_path, self.best_model_path)
            
            logger.info(f"Updated best model with win ratio {win_ratio:.2f} >= {self.update_threshold}")
            return True
        else:
            logger.info(f"Kept best model with win ratio {win_ratio:.2f} < {self.update_threshold}")
            return False
            
    def run(self):
        """Run the AlphaZero training process."""
        logger.info("Starting AlphaZero training process")
        
        for iteration in range(self.num_iterations):
            logger.info(f"Starting iteration {iteration+1}/{self.num_iterations}")
            
            # Step 1: Self-play with the best model
            self.self_play(self.best_model_path)
            
            # Step 2: Train the model
            self.train()
            
            # Step 3: Evaluate the model
            win_ratio = self.evaluate(self.current_model_path, self.best_model_path)
            
            # Step 4: Update the best model if needed
            self.update_best_model(win_ratio)
            
            logger.info(f"Completed iteration {iteration+1}/{self.num_iterations}")
            
        logger.info("AlphaZero training process completed")

class AlphaZeroPlayer:
    """
    Player that uses AlphaZero's neural network and MCTS to make moves.
    """
    def __init__(self, game, model_path, num_simulations=800, num_threads=1):
        """
        Initialize the AlphaZero player.
        
        Args:
            game: Game instance
            model_path: Path to the neural network model
            num_simulations: Number of MCTS simulations per move
            num_threads: Number of parallel threads for MCTS
        """
        self.game = game
        
        # Load the model
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
            num_threads=num_threads
        )
        
        # Root node for tree reuse
        self.root = None
        
    def reset(self):
        """Reset the player for a new game."""
        self.root = None
        
    def play(self, board, player):
        """
        Make a move on the board.
        
        Args:
            board: Board state
            player: Current player (1 for Black, -1 for White)
            
        Returns:
            action: Selected action
        """
        # Check if there are valid moves
        valid_moves = self.game.getValidMoves(board, player)
        if np.sum(valid_moves) == 0:
            # No valid moves
            return -1
        
        # Get canonical form of the board
        canonical_board = self.game.getCanonicalForm(board, player)
        
        # Select action using MCTS with the valid moves mask
        action = self.mcts.select_action(canonical_board, 1, valid_moves=valid_moves, temperature=0)
        
        # Verify the selected action is valid before returning it
        if valid_moves[action] != 1:
            logger.warning(f"MCTS selected invalid move {action}, selecting a random valid move instead")
            # Fallback to selecting a random valid move
            valid_indices = np.where(valid_moves == 1)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                return -1  # No valid moves found
        
        # Reuse tree for the next move to improve efficiency
        if self.root is not None:
            self.root = self.mcts.reuse_tree(self.root, board, player, action)
        else:
            # Create a new root node for the first move
            from .mcts import Node
            self.root = Node(self.game)
        
        return action
        
    def notify(self, board, action):
        """
        Notify the player of an opponent's move.
        
        Args:
            board: Board state before the move
            action: Action taken by the opponent
        """
        # Reuse tree after opponent's move
        if self.root is not None:
            self.root = self.mcts.reuse_tree(self.root, board, -1, action)


def main():
    """Main function to run AlphaZero training."""
    parser = argparse.ArgumentParser(description='AlphaZero for Yin-Yang')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per iteration')
    parser.add_argument('--simulations', type=int, default=800, help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per iteration')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for self-play')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory for models')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for training data')
    
    args = parser.parse_args()
    
    # Import game after parsing arguments to avoid circular imports
    from src.yin_yang import YinYangGame
    
    # Create game instance
    game = YinYangGame()
    
    # Create AlphaZero instance
    alphazero = AlphaZero(
        game=game,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        num_iterations=args.iterations,
        num_episodes=args.episodes,
        num_simulations=args.simulations,
        num_epochs=args.epochs,
        num_workers=args.workers
    )
    
    # Run the training process
    alphazero.run()

if __name__ == "__main__":
    main() 