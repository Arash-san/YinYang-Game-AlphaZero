import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(
    filename='neural_network.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YinYangNN')

class ResidualBlock(nn.Module):
    """
    Residual block for the neural network.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class YinYangNeuralNetwork(nn.Module):
    """
    Neural network for the Yin-Yang game.
    """
    def __init__(self, game, num_channels=128, num_res_blocks=10):
        super(YinYangNeuralNetwork, self).__init__()
        self.game = game
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        # Input representation: 3 channels (empty, black, white) + 
        # 2 channels for row/column completeness
        self.input_channels = 5  
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_size[0] * self.board_size[1], 
                                   self.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * self.board_size[0] * self.board_size[1], 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Log network architecture
        self._log_architecture(num_channels, num_res_blocks)
        
        # Initialize weights
        self._initialize_weights()
        
    def _log_architecture(self, num_channels, num_res_blocks):
        """Log the network architecture"""
        logger.info("YinYang Neural Network Architecture:")
        logger.info(f"Board size: {self.board_size}")
        logger.info(f"Action size: {self.action_size}")
        logger.info(f"Input channels: {self.input_channels}")
        logger.info(f"Number of filters: {num_channels}")
        logger.info(f"Number of residual blocks: {num_res_blocks}")
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        logger.info("Weights initialized with Xavier/Glorot initialization")

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Batch of board states
            
        Returns:
            policy_logits: Policy output (probability distribution over actions)
            value: Value output (evaluation of current position)
        """
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * self.board_size[0] * self.board_size[1])
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * self.board_size[0] * self.board_size[1])
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

    def predict(self, board):
        """
        Predict the policy and value for a given board state.
        
        Args:
            board: Board state
            
        Returns:
            policy: Policy output (probability distribution over actions)
            value: Value output (evaluation of current position)
        """
        # Set model to evaluation mode
        self.eval()
        
        # Convert board to input tensor
        x = self.board_to_input(board)
        
        # Add batch dimension and move to device
        x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            
            # Apply softmax to policy logits
            policy = F.softmax(policy_logits, dim=1)
            
        return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def board_to_input(self, board):
        """
        Convert board state to neural network input.
        
        Args:
            board: Board state from the game
            
        Returns:
            x: Input tensor for the neural network (5 channels)
        """
        # Get board dimensions
        n, m = self.board_size
        
        # Initialize input tensor with 5 channels
        # Channel 0: Empty spaces (1 where empty, 0 otherwise)
        # Channel 1: Black pieces (1 where black pieces are, 0 otherwise)
        # Channel 2: White pieces (1 where white pieces are, 0 otherwise)
        # Channel 3: Row completeness (fraction of filled spaces in each row)
        # Channel 4: Column completeness (fraction of filled spaces in each column)
        x = torch.zeros((self.input_channels, n, m))
        
        # Fill in the first 3 channels based on board state
        board_arr = board.get_board()
        
        x[0] = torch.tensor((board_arr == 0).astype(np.float32))  # Empty spaces
        x[1] = torch.tensor((board_arr == 1).astype(np.float32))  # Black pieces
        x[2] = torch.tensor((board_arr == -1).astype(np.float32))  # White pieces
        
        # Calculate row completeness
        for i in range(n):
            filled_count = np.sum(board_arr[i] != 0)
            completeness = filled_count / m
            x[3, i, :] = completeness
        
        # Calculate column completeness
        for j in range(m):
            filled_count = np.sum(board_arr[:, j] != 0)
            completeness = filled_count / n
            x[4, :, j] = completeness
        
        return x
    
    def save_model(self, filename):
        """
        Save the model to a file.
        
        Args:
            filename: Name of the file to save the model
        """
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'state_dict': self.state_dict(),
            'board_size': self.board_size,
            'action_size': self.action_size,
        }, filename)
        
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load the model from a file.
        
        Args:
            filename: Name of the file to load the model from
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
            
        checkpoint = torch.load(filename, map_location='cpu')
        
        # Verify board size and action size match
        if self.board_size != checkpoint['board_size']:
            logger.warning(f"Board size mismatch: expected {self.board_size}, got {checkpoint['board_size']}")
            
        if self.action_size != checkpoint['action_size']:
            logger.warning(f"Action size mismatch: expected {self.action_size}, got {checkpoint['action_size']}")
        
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {filename}") 