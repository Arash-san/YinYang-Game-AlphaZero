import numpy as np
from collections import deque

class YinYangLogic:
    """
    Logic for the Yin-Yang two-player game.
    
    Board representation:
    - 0: Empty cell
    - 1: Black piece (Player 1)
    - -1: White piece (Player 2)
    """
    
    def __init__(self, n=8, m=8):
        """Initialize a board with dimensions n x m."""
        self.n = n  # rows
        self.m = m  # columns
        self.board = np.zeros((n, m), dtype=np.int8)
        
    def get_board(self):
        """Return the current board state."""
        return self.board.copy()
    
    def place_piece(self, x, y, piece):
        """Place a piece on the board at position (x, y)."""
        if self.is_valid_move(x, y, piece):
            self.board[x, y] = piece
            return True
        return False
    
    def is_valid_move(self, x, y, piece):
        """Check if placing a piece at (x, y) is valid."""
        # Check if the position is on the board
        if not (0 <= x < self.n and 0 <= y < self.m):
            return False
        
        # Check if the position is empty
        if self.board[x, y] != 0:
            return False
        
        # Place the piece temporarily to check constraints
        self.board[x, y] = piece
        
        # Check connectivity constraint
        if not self._check_connectivity(piece):
            self.board[x, y] = 0  # Revert the move
            return False
        
        # Check 2x2 constraint
        if not self._check_2x2_constraint():
            self.board[x, y] = 0  # Revert the move
            return False
        
        # Revert the move until officially placed
        self.board[x, y] = 0
        return True
    
    def _check_connectivity(self, piece):
        """
        Check if all pieces of the same color form a single connected component.
        Uses breadth-first search to verify connectivity.
        """
        if np.sum(self.board == piece) == 0:
            return True  # No pieces of this color yet
        
        # Find the first piece of the given color
        positions = np.argwhere(self.board == piece)
        if len(positions) == 0:
            return True  # No pieces of this color
        
        start_pos = tuple(positions[0])
        
        # BFS to find all connected pieces
        visited = set([start_pos])
        queue = deque([start_pos])
        
        while queue:
            x, y = queue.popleft()
            
            # Check all four adjacent positions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check if the position is on the board
                if not (0 <= nx < self.n and 0 <= ny < self.m):
                    continue
                
                # Check if the position has a piece of the same color and hasn't been visited
                if self.board[nx, ny] == piece and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Check if all pieces of the given color have been visited
        return len(visited) == np.sum(self.board == piece)
    
    def _check_2x2_constraint(self):
        """
        Check if there are no 2x2 areas containing only pieces of the same color.
        """
        # Iterate through all possible 2x2 squares
        for i in range(self.n - 1):
            for j in range(self.m - 1):
                square = self.board[i:i+2, j:j+2]
                
                # Check if all four cells are the same color (and not empty)
                if square[0, 0] != 0 and np.all(square == square[0, 0]):
                    return False
        
        return True
    
    def get_valid_moves(self, piece):
        """Return a list of all valid moves for the given piece color."""
        valid_moves = []
        
        for i in range(self.n):
            for j in range(self.m):
                if self.is_valid_move(i, j, piece):
                    valid_moves.append((i, j))
        
        return valid_moves
    
    def has_valid_move(self, piece):
        """Check if there are any valid moves for the given piece color."""
        for i in range(self.n):
            for j in range(self.m):
                if self.is_valid_move(i, j, piece):
                    return True
        return False
    
    def count_pieces(self):
        """Count the number of pieces for each player."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        return black_count, white_count 