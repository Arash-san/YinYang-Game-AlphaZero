import numpy as np
import random


class RandomPlayer:
    """
    Random player for the Yin-Yang game.
    Makes a random valid move each turn.
    """
    
    def __init__(self, game):
        self.game = game
        
    def play(self, board, player):
        """
        Play a move based on the board state.
        
        Args:
            board: current board
            player: current player (1 for black, -1 for white)
            
        Returns:
            action: selected action
        """
        valid_moves = self.game.getValidMoves(board, player)
        valid_indices = np.where(valid_moves == 1)[0]
        
        if len(valid_indices) == 0:
            # No valid moves, return invalid action
            return -1
            
        # Select a random valid move
        action = random.choice(valid_indices)
        
        # Print the move for visibility
        x, y = self.game._action_to_coords(action)
        col = chr(ord('a') + y)
        row = x + 1
        player_name = "Black" if player == 1 else "White"
        print(f"{player_name} plays {col}{row}")
        
        return action

class HumanYinYangPlayer:
    """
    Human player for the Yin-Yang game.
    
    Input is expected as coordinates, e.g., 'a1' (column, row).
    The columns are labeled a, b, c, ... and rows are labeled 1, 2, 3, ...
    """
    
    def __init__(self, game):
        self.game = game
        
    def play(self, board, player):
        """
        Play a move based on user input.
        
        Args:
            board: current board
            player: current player (1 for black, -1 for white)
            
        Returns:
            action: selected action
        """
        valid_moves = self.game.getValidMoves(board, player)
        
        # Print board for reference
        self.game.display(board)
        
        # Show player's turn
        player_name = "Black" if player == 1 else "White"
        print(f"{player_name}'s turn")
        
        # Keep asking for input until a valid move is given
        while True:
            try:
                move = input("Enter your move (e.g., 'a1'): ").strip().lower()
                
                if len(move) < 2:
                    print("Invalid format. Please use the format 'a1'.")
                    continue
                
                # Convert input to coordinates
                col = ord(move[0]) - ord('a')
                row = int(move[1:]) - 1
                
                # Convert coordinates to action
                action = self.game._coords_to_action(row, col)
                
                # Check if the move is valid
                if 0 <= action < len(valid_moves) and valid_moves[action] == 1:
                    return action
                else:
                    print("Invalid move. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
                print("Invalid input. Please use the format 'a1'.") 