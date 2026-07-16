from enum import IntEnum
from typing import Optional, Dict, Any
from game.board import Board
from game.pieces import BagRandomizer, Tetromino
from game.collision import check_collision

class Action(IntEnum):
    """Discrete action space for the RL agent and keyboard mapper."""
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    SOFT_DROP = 3
    HARD_DROP = 4
    ROTATE_CW = 5
    ROTATE_CCW = 6
    HOLD = 7

class TetrisGame:
    """The central orchestrator for the Tetris simulation."""
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        """Resets the game state for a new episode."""
        self.board = Board()
        self.bag = BagRandomizer()
        
        self.current_piece: Tetromino = Tetromino(self.bag.pop())
        self.hold_piece: Optional[str] = None
        
        self.score: int = 0
        self.level: int = 1
        self.lines_cleared_total: int = 0
        
        self.game_over: bool = False
        self.can_hold: bool = True  # Can only hold once per piece drop
        
    def step(self, action: Action) -> Dict[str, Any]:
        """
        Advances the game state by one action.
        Returns a dictionary of step metadata (useful for reward shaping).
        """
        if self.game_over:
            return {"game_over": True, "lines_cleared": 0, "locked": False}

        lines_cleared_this_step = 0
        piece_locked = False

        if action == Action.LEFT:
            if not check_collision(self.current_piece, self.board.grid, offset_x=-1):
                self.current_piece.x -= 1
                
        elif action == Action.RIGHT:
            if not check_collision(self.current_piece, self.board.grid, offset_x=1):
                self.current_piece.x += 1
                
        elif action == Action.ROTATE_CW or action == Action.ROTATE_CCW:
            # Save original state in case rotation fails
            prev_rotation = self.current_piece.rotation
            self.current_piece.rotate(clockwise=(action == Action.ROTATE_CW))
            
            # Note: A full implementation would use SRS wall kicks here. 
            # For brevity, we simply revert if the basic rotation collides.
            if check_collision(self.current_piece, self.board.grid):
                self.current_piece.rotation = prev_rotation

        elif action == Action.HOLD:
            if self.can_hold:
                self._swap_hold_piece()

        elif action == Action.SOFT_DROP:
            if not check_collision(self.current_piece, self.board.grid, offset_y=1):
                self.current_piece.y += 1
                self.score += 1  # 1 point per soft drop cell
            else:
                lines_cleared_this_step = self._lock_and_spawn()
                piece_locked = True

        elif action == Action.HARD_DROP:
            ghost_y = self.board.get_ghost_y(self.current_piece)
            drop_distance = ghost_y - self.current_piece.y
            self.score += drop_distance * 2  # 2 points per hard drop cell
            self.current_piece.y = ghost_y
            lines_cleared_this_step = self._lock_and_spawn()
            piece_locked = True

        # Apply gravity if no drop action was taken (can be triggered externally by a timer)
        # For pure RL, we often let the agent learn to push down, but simulating gravity is safer.

        return {
            "game_over": self.game_over,
            "lines_cleared": lines_cleared_this_step,
            "locked": piece_locked
        }

    def apply_gravity(self) -> Dict[str, Any]:
        """Forces the piece down by 1 unit. Usually called by a game loop timer."""
        if self.game_over:
            return {"game_over": True, "lines_cleared": 0, "locked": False}
            
        if not check_collision(self.current_piece, self.board.grid, offset_y=1):
            self.current_piece.y += 1
            return {"game_over": False, "lines_cleared": 0, "locked": False}
        else:
            lines = self._lock_and_spawn()
            return {"game_over": self.game_over, "lines_cleared": lines, "locked": True}

    def _lock_and_spawn(self) -> int:
        """Locks the piece, clears lines, checks game over, and spawns the next piece."""
        self.board.lock_piece(self.current_piece)
        lines = self.board.clear_lines()
        
        self.lines_cleared_total += lines
        self.level = (self.lines_cleared_total // 10) + 1
        
        # Spawn next piece
        self.current_piece = Tetromino(self.bag.pop())
        self.can_hold = True
        
        # Game Over check: Does the newly spawned piece immediately collide?
        if check_collision(self.current_piece, self.board.grid):
            self.game_over = True
            
        return lines

    def _swap_hold_piece(self) -> None:
        """Handles the logic for the Hold queue."""
        current_name = self.current_piece.name
        
        if self.hold_piece is None:
            self.hold_piece = current_name
            self.current_piece = Tetromino(self.bag.pop())
        else:
            temp = self.hold_piece
            self.hold_piece = current_name
            self.current_piece = Tetromino(temp)
            
        self.can_hold = False