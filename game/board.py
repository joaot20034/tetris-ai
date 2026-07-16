import numpy as np
from typing import List, Tuple
from game.pieces import Tetromino
from game.collision import check_collision

# Map piece names to integers for the NumPy grid
PIECE_IDS = {name: idx + 1 for idx, name in enumerate(['I', 'J', 'L', 'O', 'S', 'T', 'Z'])}

class Board:
    """Manages the 10x20 Tetris grid."""
    
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        # 0 represents empty space. >0 represents locked pieces.
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        
    def lock_piece(self, piece: Tetromino) -> None:
        """Bakes the active piece into the static grid."""
        shape = piece.shape
        piece_id = PIECE_IDS.get(piece.name, 1)
        
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    board_x = piece.x + col_idx
                    board_y = piece.y + row_idx
                    # Ensure we don't write out of bounds (can happen if top-out occurs)
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.grid[board_y][board_x] = piece_id

    def clear_lines(self) -> int:
        """
        Removes fully filled rows, shifts everything down, and returns the clear count.
        Using NumPy boolean indexing makes this extremely fast for the RL loop.
        """
        # Find rows where all columns are non-zero
        full_rows = np.all(self.grid != 0, axis=1)
        lines_cleared = np.sum(full_rows)
        
        if lines_cleared > 0:
            # Keep only the non-full rows
            remaining_rows = self.grid[~full_rows]
            # Create new empty rows for the top
            new_empty_rows = np.zeros((lines_cleared, self.width), dtype=np.int8)
            # Stack empty rows on top of the remaining rows
            self.grid = np.vstack((new_empty_rows, remaining_rows))
            
        return int(lines_cleared)

    def get_ghost_y(self, piece: Tetromino) -> int:
        """Calculates the Y position of the ghost piece (hard drop landing spot)."""
        ghost_y = piece.y
        while not check_collision(piece, self.grid, offset_x=0, offset_y=ghost_y - piece.y + 1):
            ghost_y += 1
        return ghost_y

    def clone(self) -> 'Board':
        """Deep copies the board. Useful for RL look-ahead tree search if needed later."""
        new_board = Board(self.width, self.height)
        new_board.grid = np.copy(self.grid)
        return new_board