#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src package
from src.yin_yang import YinYangGame
from src.yin_yang.ai import YinYangNeuralNetwork, AlphaZeroTrainer

def visualize_board_prediction(board, policy, value):
    """
    Visualize the board state and the neural network's prediction.
    
    Args:
        board: The board state
        policy: Policy prediction (probability distribution over moves)
        value: Value prediction (evaluation of the position)
    """
    # Get board dimensions
    n, m = board.board.shape
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the board state
    board_arr = board.get_board()
    cmap = plt.cm.coolwarm
    ax1.imshow(board_arr, cmap=cmap, vmin=-1, vmax=1)
    
    # Add grid lines
    ax1.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Add labels
    for i in range(n):
        for j in range(m):
            if board_arr[i, j] == 1:
                ax1.text(j, i, "B", ha="center", va="center", color="white", fontsize=12)
            elif board_arr[i, j] == -1:
                ax1.text(j, i, "W", ha="center", va="center", color="black", fontsize=12)
    
    ax1.set_title(f"Board State (Value: {value:.3f})")
    
    # Plot the policy prediction
    policy_grid = np.zeros((n, m))
    for action, prob in enumerate(policy):
        if prob > 0:
            x, y = game._action_to_coords(action)
            policy_grid[x, y] = prob
    
    im = ax2.imshow(policy_grid, cmap='viridis', vmin=0, vmax=policy.max() if policy.max() > 0 else 1)
    
    # Add grid lines
    ax2.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax2.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Add probability labels
    for i in range(n):
        for j in range(m):
            if policy_grid[i, j] > 0.01:
                ax2.text(j, i, f"{policy_grid[i, j]:.2f}", ha="center", va="center", 
                        color="white" if policy_grid[i, j] > 0.5 else "black", fontsize=8)
    
    ax2.set_title("Policy Prediction")
    
    # Add a color bar
    fig.colorbar(im, ax=ax2, label="Probability")
    
    plt.tight_layout()
    plt.show()

def random_valid_move(game, board, player):
    """Make a random valid move on the board."""
    valid_moves = game.getValidMoves(board, player)
    valid_indices = np.where(valid_moves == 1)[0]
    
    if len(valid_indices) == 0:
        return None
        
    action = np.random.choice(valid_indices)
    return action

if __name__ == "__main__":
    # Create a game instance
    game = YinYangGame(n=6, m=6)
    
    # Create a neural network instance
    nnet = YinYangNeuralNetwork(game)
    
    # Create a trainer instance
    trainer = AlphaZeroTrainer(game, model_dir='models')
    
    print("Neural network architecture created.")
    print(f"Board size: {game.getBoardSize()}")
    print(f"Action size: {game.getActionSize()}")
    
    # Initialize the board
    board = game.getInitBoard()
    player = 1  # Player 1 (Black) starts
    
    # Example: Play a few random moves
    for _ in range(10):
        action = random_valid_move(game, board, player)
        if action is None:
            print("No valid moves left.")
            break
            
        # Apply the move
        board, player = game.getNextState(board, player, action)
    
    # Predict with the neural network
    print("\nPredicting with the neural network...")
    start_time = time.time()
    policy, value = nnet.predict(board)
    end_time = time.time()
    
    print(f"Prediction took {end_time - start_time:.4f} seconds")
    print(f"Value prediction: {value:.4f}")
    
    # Get the top moves
    top_actions = policy.argsort()[-5:][::-1]
    print("\nTop 5 moves:")
    for i, action in enumerate(top_actions):
        x, y = game._action_to_coords(action)
        col = chr(ord('a') + y)
        row = x + 1
        print(f"{i+1}. {col}{row}: {policy[action]:.4f}")
    
    # Visualize the prediction
    print("\nVisualizing the prediction...")
    visualize_board_prediction(board, policy, value)
    
    # Example of saving and loading the model
    print("\nSaving and loading the model...")
    model_path = os.path.join('models', 'example_model.pth.tar')
    
    # Save the model
    nnet.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Load the model
    new_nnet = YinYangNeuralNetwork(game)
    new_nnet.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Verify the loaded model gives the same predictions
    policy2, value2 = new_nnet.predict(board)
    
    # Check that the predictions are the same
    policy_match = np.allclose(policy, policy2)
    value_match = np.allclose(value, value2)
    
    print(f"Policy predictions match: {policy_match}")
    print(f"Value predictions match: {value_match}")
    
    print("\nExample completed successfully!") 