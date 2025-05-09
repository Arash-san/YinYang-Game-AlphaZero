import numpy as np
from .yin_yang_logic import YinYangLogic

class YinYangGame:
    """
    Game class implementation for the Yin-Yang puzzle game.
    This class implements the Game interface expected by the AlphaZero framework.
    """
    
    def __init__(self, n=8, m=8):
        self.n = n
        self.m = m
        self.action_size = n * m
    
    def getInitBoard(self):
        """
        Returns the initial board configuration.
        
        Returns:
            startBoard: a representation of the board
        """
        b = YinYangLogic(self.n, self.m)
        return b
    
    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n, self.m)
    
    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.action_size
    
    def getNextState(self, board, player, action):
        """
        Returns the board after applying action and the player who takes the next turn.
        
        Input:
            board: current board
            player: current player (1 for black, -1 for white)
            action: action taken by current player
            
        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays next (-player)
        """
        b = board
        x, y = self._action_to_coords(action)
        
        piece = 1 if player == 1 else -1
        b.place_piece(x, y, piece)
        
        return b, -player
    
    def getValidMoves(self, board, player):
        """
        Returns a binary vector of legal moves, 1 for legal, 0 for illegal.
        
        Input:
            board: current board
            player: current player (1 for black, -1 for white)
            
        Returns:
            validMoves: binary vector of legal moves (0 or 1)
        """
        piece = 1 if player == 1 else -1
        valid_coords = board.get_valid_moves(piece)
        valid_moves = np.zeros(self.action_size)
        
        for x, y in valid_coords:
            valid_moves[self._coords_to_action(x, y)] = 1
            
        return valid_moves
    
    def getGameEnded(self, board, player):
        """
        Returns:
            0 if game is ongoing
            1 if player has won
            -1 if player has lost
            small positive value for a draw (or, small negative value)
        """
        # Check if either player can make a valid move
        player_piece = 1 if player == 1 else -1
        opponent_piece = -player_piece
        
        player_can_move = board.has_valid_move(player_piece)
        opponent_can_move = board.has_valid_move(opponent_piece)
        
        # If neither player can move, the game is over
        if not player_can_move and not opponent_can_move:
            black_count, white_count = board.count_pieces()
            
            if black_count > white_count:
                # Player 1 (black) wins
                return 1 if player == 1 else -1
            elif white_count > black_count:
                # Player 2 (white) wins
                return 1 if player == -1 else -1
            else:
                # Draw
                return 0.0001  # Small positive value for draw
        
        # Game is still ongoing
        return 0
    
    def getCanonicalForm(self, board, player):
        """
        Returns the canonical form of the board from the perspective of the given player.
        
        Input:
            board: current board
            player: current player (1 for black, -1 for white)
            
        Returns:
            canonicalBoard: board from the perspective of the player
        """
        # For Yin-Yang, the canonical form is the original board since each player
        # always plays their own color and there's no need to flip the board perspective
        return board
    
    def getSymmetries(self, board, pi):
        """
        Returns all symmetrical board configurations and action probabilities.
        
        Input:
            board: current board
            pi: action probability vector
            
        Returns:
            symmForms: a list of [(board, pi)] where each tuple is a symmetrical form
        """
        # Reshape pi to match the board dimensions for easier transformations
        pi_board = np.reshape(pi, (self.n, self.m))
        
        # Initialize list of symmetries
        syms = []
        
        # Get the actual board state as a numpy array
        b = board.get_board()
        
        # Add all 8 symmetrical forms (4 rotations * 2 reflections)
        for i in range(1, 5):
            for j in [True, False]:
                # Rotate the board and pi
                newB = np.rot90(b, i)
                newPi = np.rot90(pi_board, i)
                
                # Flip if needed
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                
                # Create a new board instance with the transformed state
                new_board = YinYangLogic(self.n, self.m)
                new_board.board = newB
                
                # Flatten pi back to a 1D array
                syms.append((new_board, newPi.flatten()))
        
        return syms
    
    def stringRepresentation(self, board):
        """
        Returns a string representation of the board.
        
        Input:
            board: current board
            
        Returns:
            boardStr: a string representation of the board
        """
        return board.get_board().tostring()
    
    def _action_to_coords(self, action):
        """Convert an action number to board coordinates (x, y)."""
        return action // self.m, action % self.m
    
    def _coords_to_action(self, x, y):
        """Convert board coordinates (x, y) to an action number."""
        return x * self.m + y
    
    def display(self, board):
        """
        Display the current board state.
        
        Input:
            board: current board
        """
        b = board.get_board()
        
        print(' ' + ''.join([chr(97 + i) for i in range(self.m)]))
        for i in range(self.n):
            row_str = str(i+1)
            for j in range(self.m):
                if b[i, j] == 1:
                    row_str += 'B'  # Black piece
                elif b[i, j] == -1:
                    row_str += 'W'  # White piece
                else:
                    row_str += '.'  # Empty cell
            print(row_str) 