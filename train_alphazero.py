#!/usr/bin/env python
import os
import sys
import argparse
import logging
import multiprocessing as mp

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'  # Append to existing log file
)

# Add logging handler to output to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger('AlphaZeroTraining')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Yin-Yang AlphaZero Training')
    
    # Game configuration
    parser.add_argument('--rows', type=int, default=8, help='Number of rows on the board')
    parser.add_argument('--cols', type=int, default=8, help='Number of columns on the board')
    
    # Training configuration
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per iteration')
    parser.add_argument('--simulations', type=int, default=800, help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Parallelization
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for self-play')
    parser.add_argument('--mcts-threads', type=int, default=1, help='Number of threads for MCTS')
    
    # Directories
    parser.add_argument('--model-dir', type=str, default='models', help='Directory for model checkpoints')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for training data')
    
    # Run options
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest model')
    parser.add_argument('--mode', choices=['train', 'self-play', 'evaluate'], default='train',
                        help='Mode to run (train, self-play, evaluate)')
    parser.add_argument('--output-model', type=str, default='best_model.pth.tar',
                        help='Name of the output model file')
    
    return parser.parse_args()

def main():
    """Main function to run training."""
    args = parse_args()
    
    logger.info("Starting Yin-Yang AlphaZero Training")
    logger.info(f"Configuration: board={args.rows}x{args.cols}, iterations={args.iterations}, "
               f"episodes={args.episodes}, simulations={args.simulations}")
    
    # Import the game and AlphaZero implementation
    from src.yin_yang import YinYangGame
    from src.yin_yang.ai.alphazero import AlphaZero, generate_self_play_data
    
    # Create the game instance
    game = YinYangGame(n=args.rows, m=args.cols)
    
    # Create model and data directories if they don't exist
    for directory in [args.model_dir, args.data_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    if args.mode == 'train':
        # Create AlphaZero instance
        alphazero = AlphaZero(
            game=game,
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            num_iterations=args.iterations,
            num_episodes=args.episodes,
            num_simulations=args.simulations,
            num_epochs=args.epochs,
            num_workers=args.workers,
            mcts_threads=args.mcts_threads
        )
        
        # Run the training process
        alphazero.run()
        
        logger.info("Training completed!")
    
    elif args.mode == 'self-play':
        # Generate self-play data only
        model_path = os.path.join(args.model_dir, args.output_model)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        
        logger.info(f"Generating self-play data using model: {model_path}")
        
        data_file = generate_self_play_data(
            game=game,
            model_path=model_path,
            output_dir=args.data_dir,
            num_games=args.episodes,
            num_workers=args.workers,
            num_simulations=args.simulations
        )
        
        logger.info(f"Self-play data generation completed. Data saved to {data_file}")
    
    elif args.mode == 'evaluate':
        # Import the evaluation code
        from src.yin_yang.ai.mcts import MCTS
        from src.yin_yang.ai.neural_network import YinYangNeuralNetwork
        
        # Load the model
        model_path = os.path.join(args.model_dir, args.output_model)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        
        logger.info(f"Evaluating model: {model_path}")
        
        # Load the neural network
        neural_net = YinYangNeuralNetwork(game)
        neural_net.load_model(model_path)
        
        # Create MCTS
        mcts = MCTS(
            game=game,
            neural_net=neural_net,
            num_simulations=args.simulations,
            num_threads=args.mcts_threads
        )
        
        # Play a game against random player
        from src.yin_yang import RandomPlayer
        from src.yin_yang.ai.alphazero import AlphaZeroPlayer
        
        # Create players
        alphazero_player = AlphaZeroPlayer(
            game=game,
            model_path=model_path,
            num_simulations=args.simulations,
            num_threads=args.mcts_threads
        )
        
        random_player = RandomPlayer(game)
        
        # Play games
        num_games = 10
        alphazero_wins = 0
        random_wins = 0
        draws = 0
        
        for i in range(num_games):
            logger.info(f"Starting game {i+1}/{num_games}")
            
            # Reset players
            alphazero_player.reset()
            
            # Initialize board
            board = game.getInitBoard()
            player = 1  # Player 1 (Black) starts
            
            # Alternate who goes first
            if i % 2 == 0:
                first_player = alphazero_player
                second_player = random_player
                first_is_alphazero = True
            else:
                first_player = random_player
                second_player = alphazero_player
                first_is_alphazero = False
            
            # Play the game
            while True:
                # Get the player's move
                if player == 1:
                    action = first_player.play(board, player)
                else:
                    action = second_player.play(board, player)
                
                if action == -1:
                    # No valid moves, game is over
                    break
                
                # Make the move
                new_board, new_player = game.getNextState(board, player, action)
                
                # Notify the opponent
                if player == 1:
                    if hasattr(second_player, 'notify'):
                        second_player.notify(board, action)
                else:
                    if hasattr(first_player, 'notify'):
                        first_player.notify(board, action)
                
                board = new_board
                player = new_player
                
                # Check if game is over
                game_result = game.getGameEnded(board, player)
                if game_result != 0:
                    # Game is over
                    if game_result == 1:  # First player won
                        if first_is_alphazero:
                            alphazero_wins += 1
                        else:
                            random_wins += 1
                    elif game_result == -1:  # Second player won
                        if first_is_alphazero:
                            random_wins += 1
                        else:
                            alphazero_wins += 1
                    else:  # Draw
                        draws += 1
                    break
            
            # Log the result
            black_count, white_count = board.count_pieces()
            logger.info(f"Game {i+1} finished. Score: Black={black_count}, White={white_count}")
        
        # Calculate win rate
        win_rate = alphazero_wins / num_games
        
        logger.info(f"Evaluation completed. AlphaZero wins: {alphazero_wins}, "
                   f"Random wins: {random_wins}, Draws: {draws}")
        logger.info(f"AlphaZero win rate: {win_rate:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise 