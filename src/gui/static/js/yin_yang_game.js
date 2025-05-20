/**
 * Yin-Yang Game Web Implementation
 */
class YinYangGame {
    /**
     * Initialize the game with given dimensions
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     */
    constructor(rows = 8, cols = 8) {
        this.rows = rows;
        this.cols = cols;
        this.board = Array(rows).fill().map(() => Array(cols).fill(0));
        this.currentPlayer = 1; // 1 for black (player 1), -1 for white (player 2)
        this.gameOver = false;
        this.winner = null;
        this.lastConstraintViolation = null; // Track which constraint was violated
    }

    /**
     * Reset the game with optional new dimensions
     * @param {number} rows - Number of rows (optional)
     * @param {number} cols - Number of columns (optional)
     */
    reset(rows = null, cols = null) {
        this.rows = rows || this.rows;
        this.cols = cols || this.cols;
        this.board = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
        this.currentPlayer = 1;
        this.gameOver = false;
        this.winner = null;
        this.lastConstraintViolation = null;
    }

    /**
     * Make a move at the specified position
     * @param {number} row - Row index
     * @param {number} col - Column index
     * @returns {boolean} - Whether the move was successful
     */
    makeMove(row, col) {
        if (this.gameOver || !this.isValidMove(row, col)) {
            return false;
        }

        // Place the piece
        this.board[row][col] = this.currentPlayer;

        // Check if the game is over after this move
        this.checkGameOver();

        // Switch player if game is not over
        if (!this.gameOver) {
            this.currentPlayer = -this.currentPlayer;
            
            // Check if the new current player has any valid moves
            if (!this.hasValidMoves()) {
                // If no valid moves, switch back and check again
                this.currentPlayer = -this.currentPlayer;
                
                // If the original player also has no valid moves, game is over
                if (!this.hasValidMoves()) {
                    this.gameOver = true;
                    this.determineWinner();
                }
            }
        }

        return true;
    }

    /**
     * Place a specific piece at the specified position without checking game end conditions
     * Used for random piece placement
     * @param {number} row - Row index
     * @param {number} col - Column index
     * @param {number} piece - The piece to place (1 for black, -1 for white)
     * @returns {boolean} - Whether the placement was successful
     */
    placePiece(row, col, piece) {
        // Check if the position is valid
        if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
            return false;
        }

        // Check if the position is empty
        if (this.board[row][col] !== 0) {
            return false;
        }

        // Place the piece temporarily
        this.board[row][col] = piece;

        // Check connectivity constraint
        const connectedOk = this.checkConnectivityForPiece(piece);

        // Check 2x2 constraint
        const twoByTwoOk = this.check2x2Constraint();
        
        // Check row/column constraint
        const rowColOk = this.checkRowColumnConstraint();

        // If any constraint is violated, revert and return false
        if (!connectedOk || !twoByTwoOk || !rowColOk) {
            this.board[row][col] = 0;
            return false;
        }

        // Placement successful
        return true;
    }

    /**
     * Check if the game is over (no valid moves for either player)
     */
    checkGameOver() {
        // First check if the current player has any valid moves
        if (!this.hasValidMoves()) {
            // Then check if the opponent has any valid moves
            const savedPlayer = this.currentPlayer;
            this.currentPlayer = -this.currentPlayer;
            
            const opponentHasMoves = this.hasValidMoves();
            
            // Restore the current player
            this.currentPlayer = savedPlayer;
            
            if (!opponentHasMoves) {
                this.gameOver = true;
                this.determineWinner();
            }
        }
    }

    /**
     * Determine the winner based on piece counts
     */
    determineWinner() {
        const { blackCount, whiteCount } = this.getPieceCounts();
        
        if (blackCount > whiteCount) {
            this.winner = 1; // Black wins
        } else if (whiteCount > blackCount) {
            this.winner = -1; // White wins
        } else {
            this.winner = 0; // Draw
        }
    }

    /**
     * Check if the current player has any valid moves
     * @returns {boolean} - Whether the current player has any valid moves
     */
    hasValidMoves() {
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.isValidMove(row, col)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Get all valid moves for the current player
     * @returns {Array} - Array of [row, col] pairs for valid moves
     */
    getValidMoves() {
        const validMoves = [];
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.isValidMove(row, col)) {
                    validMoves.push([row, col]);
                }
            }
        }
        return validMoves;
    }

    /**
     * Check if placing a piece at the specified position is valid
     * @param {number} row - Row index
     * @param {number} col - Column index
     * @returns {boolean} - Whether the move is valid
     */
    isValidMove(row, col) {
        // Reset the last constraint violation
        this.lastConstraintViolation = null;
        
        // Check if the position is on the board
        if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
            return false;
        }

        // Check if the position is empty
        if (this.board[row][col] !== 0) {
            return false;
        }

        // Place the piece temporarily
        this.board[row][col] = this.currentPlayer;

        // Check connectivity constraint
        const connectedOk = this.checkConnectivity();
        if (!connectedOk) {
            this.lastConstraintViolation = 'connectivity';
            this.board[row][col] = 0;
            return false;
        }

        // Check 2x2 constraint
        const twoByTwoOk = this.check2x2Constraint();
        if (!twoByTwoOk) {
            this.lastConstraintViolation = '2x2';
            this.board[row][col] = 0;
            return false;
        }
        
        // Check row/column constraint
        const rowColOk = this.checkRowColumnConstraint();
        if (!rowColOk) {
            this.lastConstraintViolation = 'rowcol';
            this.board[row][col] = 0;
            return false;
        }

        // Remove the temporary piece
        this.board[row][col] = 0;

        return true;
    }

    /**
     * Check if all pieces of the current player's color form a single connected group
     * @returns {boolean} - Whether the connectivity constraint is satisfied
     */
    checkConnectivity() {
        return this.checkConnectivityForPiece(this.currentPlayer);
    }

    /**
     * Check if all pieces of a specified color form a single connected group
     * @param {number} piece - The piece color to check connectivity for (1 for black, -1 for white)
     * @returns {boolean} - Whether the connectivity constraint is satisfied
     */
    checkConnectivityForPiece(piece) {
        const visited = Array(this.rows).fill().map(() => Array(this.cols).fill(false));
        
        // Find the first piece of the specified color
        let startRow = -1, startCol = -1;
        
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.board[row][col] === piece) {
                    startRow = row;
                    startCol = col;
                    break;
                }
            }
            if (startRow !== -1) break;
        }
        
        // If there are no pieces of this color, connectivity is satisfied
        if (startRow === -1) return true;
        
        // Count total pieces of this color
        let totalPieces = 0;
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.board[row][col] === piece) {
                    totalPieces++;
                }
            }
        }
        
        // Do BFS to count connected pieces
        const queue = [[startRow, startCol]];
        visited[startRow][startCol] = true;
        let connectedPieces = 1;
        
        while (queue.length > 0) {
            const [row, col] = queue.shift();
            
            // Check all four adjacent positions
            const directions = [[0, 1], [1, 0], [0, -1], [-1, 0]];
            
            for (const [dRow, dCol] of directions) {
                const newRow = row + dRow;
                const newCol = col + dCol;
                
                // Check if the position is on the board
                if (newRow < 0 || newRow >= this.rows || newCol < 0 || newCol >= this.cols) {
                    continue;
                }
                
                // Check if the position has a piece of the same color and hasn't been visited
                if (this.board[newRow][newCol] === piece && !visited[newRow][newCol]) {
                    visited[newRow][newCol] = true;
                    queue.push([newRow, newCol]);
                    connectedPieces++;
                }
            }
        }
        
        // Check if all pieces are connected
        return connectedPieces === totalPieces;
    }

    /**
     * Check if there are no 2x2 areas containing only pieces of the same color
     * @returns {boolean} - Whether the 2x2 constraint is satisfied
     */
    check2x2Constraint() {
        for (let row = 0; row < this.rows - 1; row++) {
            for (let col = 0; col < this.cols - 1; col++) {
                const square = [
                    this.board[row][col],
                    this.board[row][col + 1],
                    this.board[row + 1][col],
                    this.board[row + 1][col + 1]
                ];
                
                // Check if all four cells have the same non-zero value
                if (square[0] !== 0 && square.every(cell => cell === square[0])) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Check if no complete row or column contains pieces of only one color
     * @returns {boolean} - Whether the row/column constraint is satisfied
     */
    checkRowColumnConstraint() {
        // Check rows
        for (let row = 0; row < this.rows; row++) {
            let hasBlack = false;
            let hasWhite = false;
            let hasEmpty = false;
            
            for (let col = 0; col < this.cols; col++) {
                if (this.board[row][col] === 1) {
                    hasBlack = true;
                } else if (this.board[row][col] === -1) {
                    hasWhite = true;
                } else {
                    hasEmpty = true;
                }
            }
            
            // If there are no empty cells and only one color, constraint is violated
            if (!hasEmpty && (!hasBlack || !hasWhite)) {
                return false;
            }
        }
        
        // Check columns
        for (let col = 0; col < this.cols; col++) {
            let hasBlack = false;
            let hasWhite = false;
            let hasEmpty = false;
            
            for (let row = 0; row < this.rows; row++) {
                if (this.board[row][col] === 1) {
                    hasBlack = true;
                } else if (this.board[row][col] === -1) {
                    hasWhite = true;
                } else {
                    hasEmpty = true;
                }
            }
            
            // If there are no empty cells and only one color, constraint is violated
            if (!hasEmpty && (!hasBlack || !hasWhite)) {
                return false;
            }
        }
        
        return true;
    }

    /**
     * Get the constraint violation position data
     * @param {number} row - Row index of the attempted move
     * @param {number} col - Column index of the attempted move
     * @returns {Array} - Array of [row, col] pairs that violate the constraint
     */
    getConstraintViolationPositions(row, col) {
        if (!this.lastConstraintViolation) return [];
        
        // Place the piece temporarily to analyze the violation
        this.board[row][col] = this.currentPlayer;
        
        let positions = [];
        
        if (this.lastConstraintViolation === 'rowcol') {
            // Check if the row is completely filled and has only one color
            let rowFilled = true;
            let rowHasBlack = false;
            let rowHasWhite = false;
            
            for (let c = 0; c < this.cols; c++) {
                if (this.board[row][c] === 0) {
                    rowFilled = false;
                } else if (this.board[row][c] === 1) {
                    rowHasBlack = true;
                } else if (this.board[row][c] === -1) {
                    rowHasWhite = true;
                }
            }
            
            // If the row is filled and has only one color, add all positions in that row
            if (rowFilled && (!rowHasBlack || !rowHasWhite)) {
                for (let c = 0; c < this.cols; c++) {
                    positions.push([row, c]);
                }
            }
            
            // Check if the column is completely filled and has only one color
            let colFilled = true;
            let colHasBlack = false;
            let colHasWhite = false;
            
            for (let r = 0; r < this.rows; r++) {
                if (this.board[r][col] === 0) {
                    colFilled = false;
                } else if (this.board[r][col] === 1) {
                    colHasBlack = true;
                } else if (this.board[r][col] === -1) {
                    colHasWhite = true;
                }
            }
            
            // If the column is filled and has only one color, add all positions in that column
            if (colFilled && (!colHasBlack || !colHasWhite)) {
                for (let r = 0; r < this.rows; r++) {
                    positions.push([r, col]);
                }
            }
        } else if (this.lastConstraintViolation === '2x2') {
            // Find the 2x2 square that violates the constraint
            for (let r = Math.max(0, row - 1); r < Math.min(this.rows - 1, row + 1); r++) {
                for (let c = Math.max(0, col - 1); c < Math.min(this.cols - 1, col + 1); c++) {
                    const square = [
                        this.board[r][c],
                        this.board[r][c + 1],
                        this.board[r + 1][c],
                        this.board[r + 1][c + 1]
                    ];
                    
                    if (square[0] !== 0 && square.every(cell => cell === square[0])) {
                        positions.push([r, c]);
                        positions.push([r, c + 1]);
                        positions.push([r + 1, c]);
                        positions.push([r + 1, c + 1]);
                    }
                }
            }
        } else if (this.lastConstraintViolation === 'connectivity') {
            // For connectivity violations, just highlight the attempted move
            positions.push([row, col]);
        }
        
        // Remove the temporary piece
        this.board[row][col] = 0;
        
        return positions;
    }

    /**
     * Get the piece counts for both players
     * @returns {Object} - Object with blackCount and whiteCount properties
     */
    getPieceCounts() {
        let blackCount = 0;
        let whiteCount = 0;
        
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.board[row][col] === 1) {
                    blackCount++;
                } else if (this.board[row][col] === -1) {
                    whiteCount++;
                }
            }
        }
        
        return { blackCount, whiteCount };
    }

    /**
     * Place random pairs of pieces on the board
     * @param {number} pairCount - Number of black/white pairs to place
     * @returns {boolean} - Whether all pairs were successfully placed
     */
    placeRandomPieces(pairCount) {
        if (pairCount <= 0) return true;
        
        // Get all available positions
        const availablePositions = [];
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (this.board[row][col] === 0) {
                    availablePositions.push([row, col]);
                }
            }
        }
        
        // We need to place pairCount black pieces and pairCount white pieces
        let blackPiecesPlaced = 0;
        let whitePiecesPlaced = 0;
        
        // Try up to 50 attempts to place all the pieces
        let attempts = 0;
        const maxAttempts = 100;
        
        while ((blackPiecesPlaced < pairCount || whitePiecesPlaced < pairCount) && attempts < maxAttempts) {
            attempts++;
            
            // Shuffle available positions
            this.shuffleArray(availablePositions);
            
            // Try to place a black piece if needed
            if (blackPiecesPlaced < pairCount) {
                for (let i = 0; i < availablePositions.length; i++) {
                    const [row, col] = availablePositions[i];
                    if (this.board[row][col] === 0 && this.placePiece(row, col, 1)) {
                        blackPiecesPlaced++;
                        availablePositions.splice(i, 1); // Remove this position from available positions
                        break;
                    }
                }
            }
            
            // Try to place a white piece if needed
            if (whitePiecesPlaced < pairCount) {
                for (let i = 0; i < availablePositions.length; i++) {
                    const [row, col] = availablePositions[i];
                    if (this.board[row][col] === 0 && this.placePiece(row, col, -1)) {
                        whitePiecesPlaced++;
                        availablePositions.splice(i, 1); // Remove this position from available positions
                        break;
                    }
                }
            }
            
            // If we couldn't place any new pieces in this iteration, break to avoid infinite loop
            if (blackPiecesPlaced + whitePiecesPlaced === 0 && attempts > 10) {
                break;
            }
        }
        
        // Check if all pieces were placed successfully
        return blackPiecesPlaced === pairCount && whitePiecesPlaced === pairCount;
    }
    
    /**
     * Shuffle an array in place using Fisher-Yates algorithm
     * @param {Array} array - The array to shuffle
     */
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
}

// UI Controller for the Yin-Yang Game
class YinYangGameUI {
    constructor() {
        this.game = null;
        this.boardElem = null;
        this.rowsInput = null;
        this.colsInput = null;
        this.randomPiecesInput = null;
        this.currentPlayerElem = null;
        this.gameStatusElem = null;
        this.blackCountElem = null;
        this.whiteCountElem = null;
        this.messageElem = null;
        
        // AI player settings
        this.playerBlackSelect = null;
        this.playerWhiteSelect = null;
        this.modelPathInput = null;
        this.modelStatusElem = null;
        this.aiStatusMessageElem = null; // For the new AI status message div
        this.isAIThinking = false;
        this.aiModelValidated = false;
        
        // Initialize the game
        this.init();
    }
    
    /**
     * Initialize the game UI
     */
    init() {
        // Create a new game
        this.game = new YinYangGame();
        
        // Get DOM elements
        this.boardElem = document.getElementById('game-board');
        this.rowsInput = document.getElementById('board-size-rows');
        this.colsInput = document.getElementById('board-size-cols');
        this.randomPiecesInput = document.getElementById('random-pieces-count');
        this.currentPlayerElem = document.querySelector('#current-player span');
        this.gameStatusElem = document.getElementById('game-status');
        this.blackCountElem = document.getElementById('black-count');
        this.whiteCountElem = document.getElementById('white-count');
        this.messageElem = document.getElementById('message');
        
        // AI player elements
        this.playerBlackSelect = document.getElementById('player-black');
        this.playerWhiteSelect = document.getElementById('player-white');
        this.modelPathInput = document.getElementById('model-path');
        this.numSimulationsInput = document.getElementById('num-simulations'); // New input for MCTS simulations
        this.modelStatusElem = document.getElementById('model-status'); // For model validation status
        this.aiStatusMessageElem = document.getElementById('ai-status-message'); // For general AI status
        
        // Settings elements
        this.setBoardSizeButton = document.getElementById('set-board-size');
        this.restartButton = document.getElementById('restart-game');
        
        // Random pieces elements
        this.applyRandomPiecesButton = document.getElementById('apply-random-pieces');
        
        // Bind events
        this.bindEvents();
        
        // Render the initial board
        this.renderBoard();
        this.updateGameInfo();
        
        // If AI is set as first player (black), make its move
        this.checkForAIMove();
    }
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        // Board click handler
        this.boardElem.addEventListener('click', (e) => {
            if (e.target.classList.contains('cell') && !this.game.gameOver) {
                const row = parseInt(e.target.dataset.row);
                const col = parseInt(e.target.dataset.col);
                this.handleCellClick(row, col);
            }
        });
        
        // Mouseover effects for valid moves
        this.boardElem.addEventListener('mouseover', (e) => {
            if (e.target.classList.contains('cell') && !this.game.gameOver) {
                const row = parseInt(e.target.dataset.row);
                const col = parseInt(e.target.dataset.col);
                
                if (this.game.isValidMove(row, col)) {
                    e.target.classList.add('valid-move');
                } else if (this.game.board[row][col] === 0) {
                    // This is an invalid move - show the constraint violation
                    e.target.classList.add('invalid');
                    
                    // Highlight the constraint violation if applicable
                    if (this.game.lastConstraintViolation) {
                        const violationPositions = this.game.getConstraintViolationPositions(row, col);
                        
                        // Add the violation class to all affected cells
                        for (const [r, c] of violationPositions) {
                            const cellElement = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
                            if (cellElement) {
                                cellElement.classList.add('constraint-violation');
                                if (this.game.lastConstraintViolation === 'rowcol') {
                                    cellElement.classList.add('rowcol-violation');
                                } else if (this.game.lastConstraintViolation === '2x2') {
                                    cellElement.classList.add('twobytwo-violation');
                                } else if (this.game.lastConstraintViolation === 'connectivity') {
                                    cellElement.classList.add('connectivity-violation');
                                }
                            }
                        }
                    }
                }
            }
        });
        
        // Mouseout to remove effects
        this.boardElem.addEventListener('mouseout', (e) => {
            if (e.target.classList.contains('cell')) {
                e.target.classList.remove('valid-move', 'invalid');
                
                // Remove all violation classes
                const cells = document.querySelectorAll('.cell');
                cells.forEach(cell => {
                    cell.classList.remove(
                        'constraint-violation', 
                        'rowcol-violation', 
                        'twobytwo-violation', 
                        'connectivity-violation'
                    );
                });
            }
        });
        
        // Set board size button
        this.setBoardSizeButton.addEventListener('click', () => {
            const rows = parseInt(this.rowsInput.value);
            const cols = parseInt(this.colsInput.value);
            
            if (rows >= 4 && rows <= 12 && cols >= 4 && cols <= 12) {
                this.restartGame(rows, cols);
                this.showMessage('Board size updated!', 'info');
            } else {
                this.showMessage('Invalid board size. Please use values between 4 and 12.', 'error');
            }
        });
        
        // Restart game button
        this.restartButton.addEventListener('click', () => {
            this.restartGame();
            this.showMessage('Game restarted!', 'info');
        });
        
        // Apply random pieces button
        this.applyRandomPiecesButton.addEventListener('click', () => {
            const pairCount = parseInt(this.randomPiecesInput.value);
            
            if (pairCount >= 0 && pairCount <= 10) {
                this.applyRandomPieces(pairCount);
            } else {
                this.showMessage('Invalid number of piece pairs. Please use values between 0 and 10.', 'error');
            }
        });
        
        // Add event listeners for AI player settings
        this.playerBlackSelect.addEventListener('change', () => {
            this.checkForAIMove();
        });
        
        this.playerWhiteSelect.addEventListener('change', () => {
            this.checkForAIMove();
        });
        
        // Add event listener for model validation
        document.getElementById('validate-model').addEventListener('click', () => {
            this.validateModel();
        });

        // Add event listener for model path input changes to trigger validation
        this.modelPathInput.addEventListener('change', () => {
            this.aiModelValidated = false; // Reset validation status on path change
            this.validateModel();
        });
        
        // Add event listener for num simulations input changes
        this.numSimulationsInput.addEventListener('change', () => {
            // Potentially re-validate or just note that AI player needs re-init
             this.aiModelValidated = false; // Reset validation as AI player params changed
             this.showAIStatus('MCTS simulations changed. Model will re-initialize on next AI move.', 'info');
        });
    }
    
    /**
     * Apply random pieces to the board and start a new game
     * @param {number} pairCount - Number of black/white pairs to place
     */
    applyRandomPieces(pairCount) {
        // Start with a fresh board
        this.restartGame();
        
        // Hide any previous messages
        this.messageElem.classList.add('hidden');
        
        if (pairCount === 0) {
            this.showMessage('Started new game with empty board!', 'info');
            return;
        }
        
        // Place random pieces
        const success = this.game.placeRandomPieces(pairCount);
        
        // Render the board with the random pieces
        this.renderBoard();
        this.updateGameInfo();
        
        // Show appropriate message
        if (success) {
            this.showMessage(`Started new game with ${pairCount} pairs of random pieces!`, 'success');
        } else {
            this.showMessage(`Could only place some of the requested random pieces.`, 'info');
        }
    }
    
    /**
     * Handle a cell click
     * @param {number} row - Row index 
     * @param {number} col - Column index
     */
    handleCellClick(row, col) {
        // Check if the game is over
        if (this.game.gameOver) {
            return;
        }
        
        // Check if AI is thinking
        if (this.isAIThinking) {
            this.showMessage('Please wait, AI is thinking...', 'info');
            return;
        }
        
        // Check if it's a human player's turn
        const isBlackHuman = this.playerBlackSelect.value === 'human';
        const isWhiteHuman = this.playerWhiteSelect.value === 'human';
        
        if ((this.game.currentPlayer === 1 && !isBlackHuman) || 
            (this.game.currentPlayer === -1 && !isWhiteHuman)) {
            this.showMessage("It's the AI's turn to play", 'info');
            return;
        }
        
        // Make a move
        const moveSuccess = this.game.makeMove(row, col);
        
        // If the move was successful
        if (moveSuccess) {
            // Clear constraint violation message
            this.showMessage('', 'info');
            
            // Update the board
            this.renderBoard();
            this.updateGameInfo();
            
            // Check if the game is over
            if (this.game.gameOver) {
                this.handleGameOver();
            } else {
                // Check if the next player is AI
                this.checkForAIMove();
            }
        } else {
            // Show constraint violation message
            if (this.game.lastConstraintViolation) {
                this.showMessage(`Invalid move: ${this.game.lastConstraintViolation}`, 'error');
                
                // Highlight constraint violation positions
                const positions = this.game.getConstraintViolationPositions(row, col);
                this.highlightViolationPositions(positions);
            } else {
                this.showMessage('Invalid move', 'error');
            }
        }
    }
    
    /**
     * Handle game over state
     */
    handleGameOver() {
        let message = 'Game over! ';
        
        if (this.game.winner === 1) {
            message += 'Black wins!';
        } else if (this.game.winner === -1) {
            message += 'White wins!';
        } else {
            message += "It's a draw!";
        }
        
        this.showMessage(message, 'success');
        this.gameStatusElem.textContent = 'Game over';
    }
    
    /**
     * Render the game board
     */
    renderBoard() {
        // Clear existing board
        this.boardElem.innerHTML = '';
        
        // Update the grid template
        this.boardElem.style.gridTemplateColumns = `repeat(${this.game.cols}, 50px)`;
        this.boardElem.style.gridTemplateRows = `repeat(${this.game.rows}, 50px)`;
        
        // Create cells
        for (let row = 0; row < this.game.rows; row++) {
            for (let col = 0; col < this.game.cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                
                // Add piece if any
                if (this.game.board[row][col] === 1) {
                    cell.classList.add('black');
                } else if (this.game.board[row][col] === -1) {
                    cell.classList.add('white');
                }
                
                this.boardElem.appendChild(cell);
            }
        }
    }
    
    /**
     * Update game information display
     */
    updateGameInfo() {
        // Update current player
        this.currentPlayerElem.textContent = this.game.currentPlayer === 1 ? 'Black' : 'White';
        
        // Update piece counts
        const { blackCount, whiteCount } = this.game.getPieceCounts();
        this.blackCountElem.textContent = blackCount;
        this.whiteCountElem.textContent = whiteCount;
        
        // Update game status
        if (!this.game.gameOver) {
            this.gameStatusElem.textContent = 'Game in progress';
        }
    }
    
    /**
     * Restart the game with optional new dimensions
     * @param {number} rows - Number of rows (optional)
     * @param {number} cols - Number of columns (optional)
     */
    restartGame(rows = null, cols = null) {
        // Get board size
        rows = rows || parseInt(this.rowsInput.value);
        cols = cols || parseInt(this.colsInput.value);
        
        // Reset the game
        this.game.reset(rows, cols);
        
        // Update the board
        this.renderBoard();
        this.updateGameInfo();
        
        // Clear any messages
        this.showMessage('', 'info');
        
        // Check if the first player is AI
        this.checkForAIMove();
    }
    
    /**
     * Show a message to the user
     * @param {string} text - Message text
     * @param {string} type - Message type ('success', 'error', or 'info')
     */
    showMessage(text, type = 'info') {
        this.messageElem.textContent = text;
        this.messageElem.className = `message ${type}`;
        this.messageElem.classList.remove('hidden');
        
        // Auto-hide info/success messages after 3 seconds, errors persist longer or until next action
        if (type === 'info' || type === 'success') {
            setTimeout(() => {
                this.messageElem.classList.add('hidden');
            }, 3000);
        } else if (type === 'error') {
             setTimeout(() => {
                this.messageElem.classList.add('hidden');
            }, 5000);
        }
    }

    /**
     * Show AI status messages (e.g., "AI is thinking...", errors)
     * @param {string} message - The message to display
     * @param {string} type - 'thinking', 'error', 'success', 'info'
     * @param {boolean} showSpinner - Whether to show the spinner
     */
    showAIStatus(message, type = 'info', showSpinner = false) {
        if (!this.aiStatusMessageElem) return;

        const messageTextElem = this.aiStatusMessageElem.querySelector('.message-text');
        const spinnerElem = this.aiStatusMessageElem.querySelector('.spinner');

        messageTextElem.textContent = message;
        this.aiStatusMessageElem.className = `ai-status ${type}`; // Reset classes and apply new type
        
        if (showSpinner) {
            spinnerElem.classList.remove('hidden');
        } else {
            spinnerElem.classList.add('hidden');
        }

        if (message) {
            this.aiStatusMessageElem.classList.remove('hidden');
        } else {
            this.aiStatusMessageElem.classList.add('hidden');
        }
    }
    
    hideAIStatus() {
        if (!this.aiStatusMessageElem) return;
        this.aiStatusMessageElem.classList.add('hidden');
        this.aiStatusMessageElem.querySelector('.message-text').textContent = '';
        this.aiStatusMessageElem.querySelector('.spinner').classList.add('hidden');
    }


    // Add these methods for AI functionality
    validateModel() {
        const modelPath = this.modelPathInput.value.trim();
        if (!modelPath) {
            this.showModelStatus('Please enter a model path', false); // Uses the old model status span
            this.showAIStatus('Please enter a model path to validate.', 'error', false);
            return;
        }
        
        this.showModelStatus('Validating...', null); // Old span
        this.showAIStatus('Validating model...', 'info', true);
        
        fetch('/api/validate_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                modelPath: modelPath
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.valid) {
                this.aiModelValidated = true;
                this.showModelStatus(`Valid model (${data.boardSize.rows}x${data.boardSize.cols})`, true); // Old span
                this.showAIStatus(`Model validated successfully for board size ${data.boardSize.rows}x${data.boardSize.cols}.`, 'success', false);
                
                // If the model has a specific board size, update the board
                if (data.boardSize && 
                    (data.boardSize.rows !== parseInt(this.rowsInput.value) || 
                     data.boardSize.cols !== parseInt(this.colsInput.value))) {
                    
                    // Ask user if they want to resize the board
                    if (confirm(`The model was trained on a ${data.boardSize.rows}x${data.boardSize.cols} board. Resize the board to match?`)) {
                        this.rowsInput.value = data.boardSize.rows;
                        this.colsInput.value = data.boardSize.cols;
                        this.restartGame(data.boardSize.rows, data.boardSize.cols);
                    }
                }
            } else {
                this.aiModelValidated = false;
                this.showModelStatus(data.message || 'Invalid model', false); // Old span
                this.showAIStatus(`Model validation failed: ${data.message || 'Invalid model'}`, 'error', false);
                 // Optionally, disable AI play if model is invalid
                // Example: document.getElementById('play-ai-button').disabled = true;
            }
        })
        .catch(error => {
            console.error('Error validating model:', error);
            this.aiModelValidated = false;
            this.showModelStatus('Error validating model', false); // Old span
            this.showAIStatus('Error validating model. Check console for details.', 'error', false);
        });
    }

    showModelStatus(message, isValid) { // This is for the small span next to the button
        this.modelStatusElem.textContent = message;
        this.modelStatusElem.className = ''; // Clear existing classes
        
        if (isValid === true) {
            this.modelStatusElem.classList.add('valid');
        } else if (isValid === false) {
            this.modelStatusElem.classList.add('invalid');
        }
    }

    checkForAIMove() {
        if (this.game.gameOver || this.isAIThinking) return;
        
        const currentPlayer = this.game.currentPlayer;
        const isBlackAI = this.playerBlackSelect.value === 'alphazero';
        const isWhiteAI = this.playerWhiteSelect.value === 'alphazero';
        
        // If current player is AI, make a move
        if ((currentPlayer === 1 && isBlackAI) || (currentPlayer === -1 && isWhiteAI)) {
            this.makeAIMove();
        } else {
            this.hideAIStatus(); // Hide AI status if it's human's turn
        }
    }

    makeAIMove() {
        if (!this.aiModelValidated) {
            this.showAIStatus('AI model is not validated. Please validate the model first.', 'error', false);
            // Try to validate the model first, then if successful, it might trigger AI move again.
            this.validateModel(); 
            return;
        }
        
        this.isAIThinking = true;
        this.showAIStatus('AI is thinking...', 'info', true); // Show spinner and message
        
        // Show thinking state on board (optional, if you have CSS for this)
        this.boardElem.classList.add('ai-thinking');
        
        // Prepare board data
        const boardData = [];
        for (let row = 0; row < this.game.rows; row++) {
            boardData.push([...this.game.board[row]]);
        }
        
        // Call the API to get AI move
        fetch('/api/ai_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                board: boardData,
                currentPlayer: this.game.currentPlayer,
                rows: this.game.rows,
                cols: this.game.cols,
                modelPath: this.modelPathInput.value.trim(),
                numSimulations: parseInt(this.numSimulationsInput.value, 10) || 100 // Ensure it's an int, default to 100
            }),
        })
        .then(response => {
            // Regardless of ok status, try to parse JSON, as server might send error details in JSON
            return response.json().then(data => ({ ok: response.ok, status: response.status, data }));
        })
        .then(({ ok, status, data }) => {
            this.boardElem.classList.remove('ai-thinking'); // Remove board dimming
            this.isAIThinking = false;
            // Do not hide AI status immediately, let the specific conditions below handle it or clear it.

            if (!ok || data.error) {
                const errorMessage = data.error || `AI request failed with status ${status}`;
                this.showAIStatus(`AI Error: ${errorMessage}`, 'error', false);
                // this.showMessage(`AI error: ${errorMessage}`, 'error'); // Also show in general message area if desired
                return;
            }
            
            if (!data.validMove) {
                this.showAIStatus(data.message || 'AI could not find a valid move.', 'info', false);
                // this.showMessage(data.message || 'AI could not find a valid move', 'warning');
                
                // Check if game is over
                if (!this.game.hasValidMoves()) {
                    this.game.gameOver = true;
                    this.game.determineWinner();
                    this.handleGameOver(); // This will show game over message
                } else {
                    // Switch player
                    this.game.currentPlayer = -this.game.currentPlayer;
                    this.updateGameInfo();
                    // AI status about not finding a move is already shown.
                    // Check if the new player (human or AI) should make a move.
                    this.checkForAIMove(); 
                }
                return;
            }
            
            this.hideAIStatus(); // Hide "AI is thinking" on successful move
            // Make the move
            const { row, col } = data;
            const moveSuccess = this.game.makeMove(row, col);
            
            if (moveSuccess) {
                this.renderBoard();
                this.updateGameInfo();
                
                // Schedule next AI move if needed
                setTimeout(() => this.checkForAIMove(), 500);
            } else {
                // This case should ideally not happen if server sends valid moves
                this.showAIStatus('AI returned an invalid move. Please report this issue.', 'error', false);
                // this.showMessage('AI returned an invalid move', 'error');
            }
        })
        .catch(error => { // Network errors or JSON parsing errors
            console.error('Error making AI move:', error);
            this.boardElem.classList.remove('ai-thinking');
            this.isAIThinking = false;
            this.showAIStatus(`Error communicating with AI: ${error.message}`, 'error', false);
            // this.showMessage(`Error communicating with AI: ${error.message}`, 'error');
        });
    }
}

// Initialize the game when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const gameUI = new YinYangGameUI();
});

// Add AI thinking CSS styles
document.addEventListener('DOMContentLoaded', function() {
    document.head.insertAdjacentHTML('beforeend', `
    <style>
        .game-board.ai-thinking {
            opacity: 0.7;
            pointer-events: none;
        }
        
        .game-cell:hover {
            cursor: pointer;
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
    `);
}); 