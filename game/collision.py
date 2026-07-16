from typing import List
from game.pieces import Tetromino
import numpy as np

def check_collision(piece: Tetromino, board_grid: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Checks if a piece at its current location + offsets collides with boundaries or locked blocks.
    
    Args:
        piece: The active Tetromino.
        board_grid: 2D numpy array representing the locked board.
        offset_x: Proposed shift in the X axis.
        offset_y: Proposed shift in the Y axis.
        
    Returns:
        True if there is a collision, False otherwise.
    """
    shape = piece.shape
    board_height, board_width = board_grid.shape
    
    for row_idx, row in enumerate(shape):
        for col_idx, cell in enumerate(row):
            if cell:  # If this part of the tetromino is solid
                
                # Calculate absolute positions on the board
                board_x = piece.x + col_idx + offset_x
                board_y = piece.y + row_idx + offset_y
                
                # Check 1: Floor and Ceiling boundaries
                if board_y >= board_height or board_y < 0:
                    return True
                
                # Check 2: Wall boundaries
                if board_x < 0 or board_x >= board_width:
                    return True
                
                # Check 3: Collision with existing locked blocks
                if board_grid[board_y][board_x] != 0:
                    return True
                    
    return False