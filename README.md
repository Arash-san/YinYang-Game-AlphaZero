# YinYang-AlphaZero

An implementation of the AlphaZero algorithm applied to the Yin-Yang puzzle game.

## About the Project

This project implements the AlphaZero algorithm to learn and play the Yin-Yang puzzle game. AlphaZero is a reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks to achieve superhuman performance in various games.

### What is Yin-Yang?

Yin-Yang is a puzzle game played on a rectangular grid. The objective is to fill the board with black (Yin) and white (Yang) pieces following these constraints:

- All pieces of the same color must be connected
- No 2x2 block can be filled with pieces of the same color
- No row or column can be filled entirely with pieces of the same color

Players take turns placing pieces, and the player with the most pieces on the board when no more valid moves are possible wins.

## Project Structure

```
YinYang-Alphazero/
  ├── alpha_zero/        # Core AlphaZero implementation components
  │   ├── checkpoints/   # Saved model checkpoints
  │   └── games/         # Self-play game records
  ├── data/              # Training data
  ├── models/            # Trained models
  ├── src/               # Source code
  │   ├── gui/           # Web interface for playing against the AI
  │   └── yin_yang/      # Yin-Yang game implementation
  ├── play_yin_yang_web.py  # Script to run the web interface
  ├── requirements.txt      # Python dependencies
  ├── run_example.py        # Example script to run the algorithm
  └── train_alphazero.py    # Script to train the AlphaZero model
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training AlphaZero

The `train_alphazero.py` script is the main entry point for training, evaluating, and generating self-play data with the AlphaZero model. It supports three different modes of operation:

#### 1. Training Mode (Default)

In training mode, the script runs the complete AlphaZero training pipeline:
- Creates a neural network model for Yin-Yang
- Generates self-play games using MCTS guided by the neural network
- Trains the neural network on the generated game data
- Iteratively improves the model through multiple cycles of self-play and training

```bash
python train_alphazero.py --rows 6 --cols 6 --iterations 100 --episodes 100
```

#### 2. Self-Play Mode

In self-play mode, the script only generates self-play games using an existing model without performing any training:

```bash
python train_alphazero.py --mode self-play --output-model best_model.pth.tar --episodes 200
```

This is useful for generating additional training data with a specific model version.

#### 3. Evaluation Mode

In evaluation mode, the script evaluates a trained model by playing against a random player:

```bash
python train_alphazero.py --mode evaluate --output-model best_model.pth.tar
```

This runs 10 games (alternating who goes first) and reports the win rate of the AlphaZero model against the random player.

#### Key Parameters

- `--rows`, `--cols`: Board dimensions (default: 8x8)
- `--iterations`: Number of training iterations (default: 100) 
- `--episodes`: Number of self-play episodes per iteration (default: 100)
- `--simulations`: Number of MCTS simulations per move (default: 800)
- `--workers`: Number of parallel workers for self-play (default: 1)
- `--mcts-threads`: Number of threads for MCTS (default: 1)
- `--resume`: Resume training from the latest model
- `--model-dir`: Directory for model checkpoints (default: "models")
- `--data-dir`: Directory for training data (default: "data")
- `--output-model`: Name of the model file to use/create (default: "best_model.pth.tar")

Run `python train_alphazero.py --help` for a complete list of options.

### Running Examples

The `run_example.py` script provides a simple way to see the trained AlphaZero model in action. This script:

1. Sets up the Python environment
2. Runs the example implementation located at `src/yin_yang/ai/example.py`
3. Demonstrates the AlphaZero algorithm playing Yin-Yang games

Running this script is a great way to quickly see the capabilities of the trained model without needing to set up the full web interface:

```bash
python run_example.py
```

The example script will show gameplay, with the board state printed to the console after each move, allowing you to observe how the AI makes decisions.

### Playing Against AlphaZero

To play against the trained AlphaZero model in a web interface:

```bash
python play_yin_yang_web.py
```

This starts a web server that you can access through your browser.

Options:
- `--host`: Host to run the server on (default: localhost)
- `--port`: Port to run the server on (default: 8000)
- `--no-browser`: Prevent automatically opening a browser

## Dependencies

- NumPy (1.24.3)
- PyTorch (2.0.1)
- TorchVision (0.15.2)
- Flask (2.0.1)
- Matplotlib (3.7.2)
- argparse (1.4.0)
- tqdm (4.66.1)
- pytest (7.4.0)

## License

This project is open source and available under the MIT License. 