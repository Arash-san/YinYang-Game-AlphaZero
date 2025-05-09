import os
import webbrowser
import json
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import logging

from src.yin_yang import YinYangGame
from src.yin_yang.ai.alphazero import AlphaZeroPlayer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('YinYangWebServer')

# Global variables
alphazero_player = None
game_instance = None

# Create Flask app
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

@app.route('/')
def index():
    """Serve the index page"""
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/api/ai_move', methods=['POST'])
def get_ai_move():
    """Get an AI move for the current board state"""
    global alphazero_player, game_instance
    
    try:
        data = request.json
        board_state = data.get('board')
        player = data.get('currentPlayer')
        rows = data.get('rows')
        cols = data.get('cols')
        model_path = data.get('modelPath', 'models/best_model.pth.tar')
        
        logger.info(f"AI move requested for board size {rows}x{cols}, player {player}, model: {model_path}")
        
        # Initialize game and AI if not already done
        if game_instance is None or game_instance.getBoardSize() != (rows, cols):
            logger.info(f"Creating new game instance with size {rows}x{cols}")
            game_instance = YinYangGame(rows, cols)
            
        if alphazero_player is None or alphazero_player.game.getBoardSize() != (rows, cols):
            try:
                logger.info(f"Initializing AlphaZero player with model: {model_path}")
                alphazero_player = AlphaZeroPlayer(
                    game=game_instance,
                    model_path=model_path,
                    num_simulations=100,
                    num_threads=1
                )
                logger.info(f"AlphaZero player initialized with board size {rows}x{cols}")
            except Exception as e:
                logger.error(f"Error initializing AlphaZero player: {str(e)}", exc_info=True)
                return jsonify({'error': f"Failed to initialize AlphaZero: {str(e)}"}), 500
        
        # Create YinYangLogic board from JavaScript board representation
        board = game_instance.getInitBoard()
        board_arr = board.get_board()
        
        # Fill the board with the current state
        for r in range(rows):
            for c in range(cols):
                board_arr[r, c] = board_state[r][c]
        
        # Log board state
        logger.info(f"Board state for AI move: \n{board_arr}")
        logger.info(f"Current player: {player}")
        
        # Check for valid moves first
        valid_moves = game_instance.getValidMoves(board, player)
        valid_move_count = np.sum(valid_moves)
        logger.info(f"Number of valid moves: {valid_move_count}")
        
        if valid_move_count == 0:
            logger.info("No valid moves available")
            return jsonify({'validMove': False, 'message': 'No valid moves available'})
        
        # Get AI move
        try:
            logger.info("Calling AlphaZero player to make a move...")
            
            # Reset the AI player's root node to ensure clean state
            alphazero_player.reset()
            
            # Get AI move
            action = alphazero_player.play(board, player)
            
            if action == -1:
                logger.info("AI found no valid moves")
                return jsonify({'validMove': False, 'message': 'No valid moves available'})
            
            # Convert action to row, col
            row, col = game_instance._action_to_coords(action)
            logger.info(f"AI selected move: ({row}, {col}) - action {action}")
            
            # Double-check that this is actually a valid move
            if valid_moves[action] != 1:
                logger.error(f"AI selected invalid move at ({row}, {col})! Trying to find any valid move...")
                
                # Find a random valid move as a fallback
                valid_indices = np.where(valid_moves == 1)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                    row, col = game_instance._action_to_coords(action)
                    logger.info(f"Selected fallback move: ({row}, {col}) - action {action}")
                else:
                    logger.error("No valid moves found after verification")
                    return jsonify({'validMove': False, 'message': 'No valid moves available'})
            
            return jsonify({
                'validMove': True,
                'row': int(row),
                'col': int(col)
            })
        except Exception as e:
            logger.error(f"Error getting AI move: {str(e)}", exc_info=True)
            return jsonify({'error': f"Failed to get AI move: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error processing AI move request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/validate_model', methods=['POST'])
def validate_model():
    """Validate that the AlphaZero model exists and get its board size"""
    try:
        data = request.json
        model_path = data.get('modelPath', 'models/best_model.pth.tar')
        
        if not os.path.exists(model_path):
            return jsonify({'valid': False, 'message': f"Model file not found: {model_path}"})
        
        # Try to load the model to get board size
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        board_size = checkpoint.get('board_size')
        
        if board_size:
            return jsonify({
                'valid': True, 
                'boardSize': {'rows': board_size[0], 'cols': board_size[1]}
            })
        else:
            return jsonify({'valid': False, 'message': "Could not determine model board size"})
    
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        return jsonify({'valid': False, 'message': str(e)})

def run_server(host='localhost', port=8000, open_browser=True):
    """
    Run the Flask server.
    
    Args:
        host: Host to run the server on (default: localhost)
        port: Port to run the server on (default: 8000)
        open_browser: Whether to open a browser (default: True)
    """
    url = f"http://{host}:{port}"
    print(f"Serving Yin-Yang game at {url}")
    
    # Open browser
    if open_browser:
        webbrowser.open(url)
    
    # Run the server
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    run_server() 