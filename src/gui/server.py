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
        num_simulations = data.get('numSimulations', 100)

        # Validate num_simulations
        try:
            num_simulations = int(num_simulations)
            if num_simulations < 10 or num_simulations > 1000: # Example range, adjust as needed
                logger.warning(f"Invalid num_simulations value: {num_simulations}. Using default 100.")
                num_simulations = 100
        except ValueError:
            logger.warning(f"Non-integer num_simulations value: {num_simulations}. Using default 100.")
            num_simulations = 100
        
        logger.info(f"AI move requested for board size {rows}x{cols}, player {player}, model: {model_path}, simulations: {num_simulations}")
        
        # Initialize game instance if not already done or if board size changed
        if game_instance is None or game_instance.getBoardSize() != (rows, cols):
            logger.info(f"Creating new game instance with size {rows}x{cols}")
            game_instance = YinYangGame(rows, cols)
            alphazero_player = None # Reset player if game changes

        # Initialize AlphaZeroPlayer if not already done, or if model/params changed
        # Check if num_simulations is part of alphazero_player and compare
        current_simulations = getattr(alphazero_player, 'num_simulations', None) if alphazero_player else None
        if (alphazero_player is None or 
            alphazero_player.model_path != model_path or 
            current_simulations != num_simulations or
            alphazero_player.game.getBoardSize() != (rows, cols)):
            try:
                logger.info(f"Validating model {model_path} for board size {rows}x{cols}")
                # Check if model file exists
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return jsonify({'error': f"Failed to load AI model: File not found: {model_path}"}), 500

                # Load model to check board size (if available in checkpoint)
                import torch
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_board_size = checkpoint.get('board_size')
                    if model_board_size and model_board_size != (rows, cols):
                        logger.error(f"Model board size {model_board_size} does not match game board size {(rows, cols)}")
                        return jsonify({'error': f"Failed to load AI model: Model board size {model_board_size} does not match game board size {(rows, cols)}"}), 500
                except (RuntimeError, torch.serialization.pickle.UnpicklingError) as e:
                    logger.error(f"Error loading model checkpoint for board size check: {str(e)}", exc_info=True)
                    return jsonify({'error': f"Failed to load AI model: Model file corrupted or invalid format: {str(e)}"}), 500
                except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
                    logger.error(f"Model file not found during torch.load: {model_path}")
                    return jsonify({'error': f"Failed to load AI model: File not found: {model_path}"}), 500
                except Exception as e: # Catch any other loading errors
                    logger.error(f"Unexpected error loading model checkpoint: {str(e)}", exc_info=True)
                    return jsonify({'error': f"Failed to load AI model: Unexpected error: {str(e)}"}), 500

                logger.info(f"Initializing AlphaZero player with model: {model_path}")
                alphazero_player = AlphaZeroPlayer(
                    game=game_instance,
                    model_path=model_path,
                    num_simulations=num_simulations, # Use from request
                    num_threads=1      # TODO: Make configurable if desired
                )
                # Store model_path and num_simulations in player object to detect changes
                alphazero_player.model_path = model_path 
                alphazero_player.num_simulations = num_simulations # Store for comparison
                logger.info(f"AlphaZero player initialized with board size {rows}x{cols}, simulations: {num_simulations}")

            except FileNotFoundError:
                logger.error(f"Model file not found: {model_path}", exc_info=True)
                return jsonify({'error': f"Failed to load AI model: File not found: {model_path}"}), 500
            except (RuntimeError, torch.serialization.pickle.UnpicklingError) as e: # PyTorch specific errors
                logger.error(f"Error initializing AlphaZero player (corrupted/incompatible model?): {str(e)}", exc_info=True)
                return jsonify({'error': f"Failed to load AI model: Model file corrupted or invalid format: {str(e)}"}), 500
            except Exception as e: # Catch any other initialization errors
                logger.error(f"Error initializing AlphaZero player: {str(e)}", exc_info=True)
                return jsonify({'error': f"Failed to load AI model: {str(e)}"}), 500
        
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
        logger.info(f"Validating model: {model_path}")

        if not os.path.exists(model_path):
            logger.warning(f"Validation failed: Model file not found: {model_path}")
            return jsonify({'valid': False, 'message': f"File not found: {model_path}"})
        
        # Try to load the model to get board size
        import torch
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            board_size = checkpoint.get('board_size')
            
            if board_size and isinstance(board_size, (list, tuple)) and len(board_size) == 2:
                logger.info(f"Model validated successfully. Board size: {board_size}")
                return jsonify({
                    'valid': True, 
                    'boardSize': {'rows': board_size[0], 'cols': board_size[1]}
                })
            else:
                logger.warning(f"Validation failed: Could not determine model board size from checkpoint: {model_path}")
                return jsonify({'valid': False, 'message': "Could not determine model board size from model file."})
        
        except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
            logger.error(f"Validation failed: Model file not found during torch.load: {model_path}", exc_info=True)
            return jsonify({'valid': False, 'message': f"File not found: {model_path}"})
        except (RuntimeError, torch.serialization.pickle.UnpicklingError) as e:
            logger.error(f"Validation failed: Model corrupted or invalid format ({model_path}): {str(e)}", exc_info=True)
            return jsonify({'valid': False, 'message': f"Model corrupted or invalid format: {str(e)}"})
        except Exception as e: # Catch any other loading errors
            logger.error(f"Validation failed: Unexpected error loading model ({model_path}): {str(e)}", exc_info=True)
            return jsonify({'valid': False, 'message': f"Unexpected error: {str(e)}"})
            
    except Exception as e: # Catch errors like request.json failing
        logger.error(f"Error processing validate_model request: {str(e)}", exc_info=True)
        return jsonify({'valid': False, 'message': f"Error processing request: {str(e)}"}), 400

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