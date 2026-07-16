import pygame
import numpy as np
from typing import Tuple, Dict
from game.game import TetrisGame
from game.pieces import COLORS, SHAPES
from game.board import PIECE_IDS

# Create a reverse lookup for colors based on the board's integer IDs
ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    idx: COLORS[name] for name, idx in PIECE_IDS.items()
}
ID_TO_COLOR[0] = (20, 20, 20)  # Very dark gray for empty background cells

class Renderer:
    """
    Stateless Pygame renderer. 
    Only initializes Pygame if a window is explicitly requested, 
    allowing headless RL training.
    """
    
    def __init__(self, cell_size: int = 30):
        self.cell_size = cell_size
        
        # UI Layout dimensions
        self.board_width = 10 * cell_size
        self.board_height = 20 * cell_size
        self.sidebar_width = 200
        
        self.screen_width = self.board_width + (self.sidebar_width * 2)
        self.screen_height = self.board_height + (cell_size * 2)
        
        self.window: pygame.Surface = None
        self.font: pygame.font.Font = None
        self.is_initialized: bool = False

    def _init_pygame(self) -> None:
        """Lazy initialization of Pygame."""
        pygame.init()
        pygame.display.set_caption("Tetris AI - Training Environment")
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.SysFont('consolas', 24)
        self.small_font = pygame.font.SysFont('consolas', 18)
        self.is_initialized = True

    def render(self, game: TetrisGame) -> None:
        """Paints the entire game state to the screen."""
        if not self.is_initialized:
            self._init_pygame()

        self.window.fill((10, 10, 10))  # Black background
        
        self._draw_board(game)
        self._draw_active_piece(game)
        self._draw_ghost_piece(game)
        self._draw_ui(game)
        
        pygame.display.flip()

    def _draw_board(self, game: TetrisGame) -> None:
        """Draws the locked grid."""
        board_offset_x = self.sidebar_width
        board_offset_y = self.cell_size
        
        # Draw border
        pygame.draw.rect(
            self.window, 
            (100, 100, 100), 
            (board_offset_x - 2, board_offset_y - 2, self.board_width + 4, self.board_height + 4), 
            2
        )

        # Draw grid cells
        for y in range(game.board.height):
            for x in range(game.board.width):
                cell_val = game.board.grid[y][x]
                color = ID_TO_COLOR[cell_val]
                
                rect = (
                    board_offset_x + (x * self.cell_size),
                    board_offset_y + (y * self.cell_size),
                    self.cell_size - 1,
                    self.cell_size - 1
                )
                pygame.draw.rect(self.window, color, rect)

    def _draw_active_piece(self, game: TetrisGame) -> None:
        """Draws the currently falling Tetromino."""
        if game.game_over:
            return
            
        piece = game.current_piece
        board_offset_x = self.sidebar_width
        board_offset_y = self.cell_size
        
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = (
                        board_offset_x + ((piece.x + col_idx) * self.cell_size),
                        board_offset_y + ((piece.y + row_idx) * self.cell_size),
                        self.cell_size - 1,
                        self.cell_size - 1
                    )
                    pygame.draw.rect(self.window, piece.color, rect)

    def _draw_ghost_piece(self, game: TetrisGame) -> None:
        """Draws the transparent landing indicator."""
        if game.game_over:
            return
            
        piece = game.current_piece
        ghost_y = game.board.get_ghost_y(piece)
        
        board_offset_x = self.sidebar_width
        board_offset_y = self.cell_size
        
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = (
                        board_offset_x + ((piece.x + col_idx) * self.cell_size),
                        board_offset_y + ((ghost_y + row_idx) * self.cell_size),
                        self.cell_size - 1,
                        self.cell_size - 1
                    )
                    # Draw only the outline for the ghost piece
                    pygame.draw.rect(self.window, COLORS['GHOST'], rect, 2)

    def _draw_ui(self, game: TetrisGame) -> None:
        """Draws the score, level, hold queue, and next queue panels."""
        # Left Panel (Hold & Stats)
        hold_text = self.font.render("HOLD", True, (255, 255, 255))
        self.window.blit(hold_text, (50, 50))
        
        if game.hold_piece:
            self._draw_minipiece(game.hold_piece, 50, 90)

        stats_y = 250
        stats = [
            f"SCORE: {game.score}",
            f"LEVEL: {game.level}",
            f"LINES: {game.lines_cleared_total}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, (200, 200, 200))
            self.window.blit(text, (20, stats_y + (i * 30)))

        # Right Panel (Next Queue)
        next_text = self.font.render("NEXT", True, (255, 255, 255))
        self.window.blit(next_text, (self.screen_width - 150, 50))
        
        next_queue = game.bag.get_queue(4)
        for i, piece_name in enumerate(next_queue):
            self._draw_minipiece(piece_name, self.screen_width - 150, 90 + (i * 90))

    def _draw_minipiece(self, piece_name: str, x: int, y: int) -> None:
        """Helper to draw pieces in the Hold and Next panels."""
        shape = SHAPES[piece_name][0]
        color = COLORS[piece_name]
        mini_size = self.cell_size // 1.5
        
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = (
                        x + (col_idx * mini_size),
                        y + (row_idx * mini_size),
                        mini_size - 1,
                        mini_size - 1
                    )
                    pygame.draw.rect(self.window, color, rect)

    def close(self) -> None:
        """Cleans up Pygame resources."""
        if self.is_initialized:
            pygame.quit()
            self.is_initialized = False