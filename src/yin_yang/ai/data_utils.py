import numpy as np
import torch
import logging

logger = logging.getLogger('YinYangNN.data')

class DataProcessor:
    """
    Data preprocessing and augmentation for neural network training.
    """
    def __init__(self, game):
        self.game = game
        self.board_size = game.getBoardSize()
        logger.info(f"DataProcessor initialized with board size {self.board_size}")
        
    def preprocess_sample(self, board, policy, player):
        """
        Preprocess a single training sample.
        
        Args:
            board: Current board state
            policy: Policy vector (probability distribution over moves)
            player: Current player (1 for Black, -1 for White)
            
        Returns:
            board_tensor: Processed board tensor
            policy_tensor: Processed policy tensor
        """
        # Convert board to input tensor using the board_to_input method
        from .neural_network import YinYangNeuralNetwork
        dummy_nn = YinYangNeuralNetwork(self.game)
        board_tensor = dummy_nn.board_to_input(board)
        
        # Convert policy vector to tensor
        policy_tensor = torch.FloatTensor(policy)
        
        return board_tensor, policy_tensor
        
    def augment_sample(self, board, policy):
        """
        Apply data augmentation to a training sample using rotations and reflections.
        
        Args:
            board: Board state tensor (channels, height, width)
            policy: Policy vector
            
        Returns:
            augmented_samples: List of augmented (board, policy) pairs
        """
        augmented_samples = []
        
        # Original board and policy
        augmented_samples.append((board, policy))
        
        # Get board dimensions
        n, m = self.board_size
        
        # Convert the policy vector to a 2D grid for easier rotation/reflection
        policy_grid = self._policy_to_grid(policy)
        
        # Apply rotations and reflections
        for rot in range(1, 4):  # 90, 180, 270 degrees rotations
            # Rotate board
            rotated_board = torch.zeros_like(board)
            for c in range(board.shape[0]):
                rotated_board[c] = torch.rot90(board[c], rot)
            
            # Rotate policy grid
            rotated_policy_grid = np.rot90(policy_grid, rot)
            rotated_policy = self._grid_to_policy(rotated_policy_grid)
            
            # Convert numpy array to torch tensor
            if isinstance(rotated_policy, np.ndarray):
                rotated_policy = torch.FloatTensor(rotated_policy)
                
            augmented_samples.append((rotated_board, rotated_policy))
        
        # Horizontal flip
        flipped_board = torch.zeros_like(board)
        for c in range(board.shape[0]):
            flipped_board[c] = torch.flip(board[c], [1])  # Flip horizontally
        
        flipped_policy_grid = np.flip(policy_grid, 1)  # Flip horizontally
        flipped_policy = self._grid_to_policy(flipped_policy_grid)
        
        # Convert numpy array to torch tensor
        if isinstance(flipped_policy, np.ndarray):
            flipped_policy = torch.FloatTensor(flipped_policy)
            
        augmented_samples.append((flipped_board, flipped_policy))
        
        # Vertical flip
        flipped_board_v = torch.zeros_like(board)
        for c in range(board.shape[0]):
            flipped_board_v[c] = torch.flip(board[c], [0])  # Flip vertically
        
        flipped_policy_grid_v = np.flip(policy_grid, 0)  # Flip vertically
        flipped_policy_v = self._grid_to_policy(flipped_policy_grid_v)
        
        # Convert numpy array to torch tensor
        if isinstance(flipped_policy_v, np.ndarray):
            flipped_policy_v = torch.FloatTensor(flipped_policy_v)
            
        augmented_samples.append((flipped_board_v, flipped_policy_v))
        
        # Diagonal flips (combined with rotations)
        flipped_board_d1 = torch.zeros_like(board)
        for c in range(board.shape[0]):
            flipped_board_d1[c] = torch.transpose(board[c], 0, 1)  # Transpose
        
        flipped_policy_grid_d1 = np.transpose(policy_grid)  # Transpose
        flipped_policy_d1 = self._grid_to_policy(flipped_policy_grid_d1)
        
        # Convert numpy array to torch tensor
        if isinstance(flipped_policy_d1, np.ndarray):
            flipped_policy_d1 = torch.FloatTensor(flipped_policy_d1)
            
        augmented_samples.append((flipped_board_d1, flipped_policy_d1))
        
        # Second diagonal flip
        flipped_board_d2 = torch.zeros_like(board)
        for c in range(board.shape[0]):
            flipped_board_d2[c] = torch.flip(torch.transpose(board[c], 0, 1), [0, 1])  # Transpose + flip both axes
        
        flipped_policy_grid_d2 = np.flip(np.transpose(policy_grid), (0, 1))  # Transpose + flip both axes
        flipped_policy_d2 = self._grid_to_policy(flipped_policy_grid_d2)
        
        # Convert numpy array to torch tensor
        if isinstance(flipped_policy_d2, np.ndarray):
            flipped_policy_d2 = torch.FloatTensor(flipped_policy_d2)
            
        augmented_samples.append((flipped_board_d2, flipped_policy_d2))
        
        return augmented_samples
    
    def _policy_to_grid(self, policy):
        """
        Convert policy vector to 2D grid.
        
        Args:
            policy: Policy vector (probability distribution over moves)
            
        Returns:
            policy_grid: 2D grid representation of policy
        """
        n, m = self.board_size
        policy_grid = np.zeros((n, m))
        
        # Convert policy to numpy array if it's a tensor
        if isinstance(policy, torch.Tensor):
            policy = policy.detach().cpu().numpy()
            
        for action in range(len(policy)):
            if policy[action] > 0:
                x, y = self.game._action_to_coords(action)
                policy_grid[x, y] = policy[action]
                
        return policy_grid
    
    def _grid_to_policy(self, policy_grid):
        """
        Convert 2D grid back to policy vector.
        
        Args:
            policy_grid: 2D grid representation of policy
            
        Returns:
            policy: Policy vector (as torch.FloatTensor)
        """
        n, m = self.board_size
        policy = np.zeros(n * m)
        
        for x in range(n):
            for y in range(m):
                if policy_grid[x, y] > 0:
                    action = self.game._coords_to_action(x, y)
                    policy[action] = policy_grid[x, y]
        
        # Convert to torch tensor
        return torch.FloatTensor(policy)

def create_dataset_from_games(game_data, game, augment=True):
    """
    Create a dataset from a list of games.
    
    Args:
        game_data: List of (board, policy, value) tuples
        game: Game instance
        augment: Whether to apply data augmentation
        
    Returns:
        board_tensors: List of board tensors
        policy_tensors: List of policy tensors
        value_tensors: List of value tensors
    """
    processor = DataProcessor(game)
    
    board_tensors = []
    policy_tensors = []
    value_tensors = []
    
    for board, policy, value in game_data:
        board_tensor, policy_tensor = processor.preprocess_sample(board, policy, 1)  # Assuming player 1 perspective
        
        if augment:
            # Apply data augmentation
            augmented_samples = processor.augment_sample(board_tensor, policy_tensor)
            
            for aug_board, aug_policy in augmented_samples:
                board_tensors.append(aug_board)
                policy_tensors.append(aug_policy)
                value_tensors.append(torch.FloatTensor([value]))
        else:
            board_tensors.append(board_tensor)
            policy_tensors.append(policy_tensor)
            value_tensors.append(torch.FloatTensor([value]))
    
    logger.info(f"Created dataset with {len(board_tensors)} samples (augmentation={augment})")
    return board_tensors, policy_tensors, value_tensors 