import numpy as np
from typing import Dict, Any
from ai.config import RLConfig

class RewardShaper:
    """Calculates heuristic penalties and rewards for the RL agent."""
    
    def __init__(self, config: RLConfig):
        self.config = config

    def get_column_heights(self, grid: np.ndarray) -> np.ndarray:
        """Returns an array of heights for each column."""
        # Find the first non-zero element in each column
        is_filled = grid != 0
        # argmax returns the first index of True. If a column is empty, it returns 0.
        # We need to handle completely empty columns specifically.
        heights = np.zeros(grid.shape[1], dtype=np.int32)
        for col in range(grid.shape[1]):
            filled_indices = np.where(is_filled[:, col])[0]
            if len(filled_indices) > 0:
                heights[col] = grid.shape[0] - filled_indices[0]
        return heights

    def get_bumpiness(self, heights: np.ndarray) -> int:
        """Calculates the sum of absolute differences between adjacent columns."""
        return int(np.sum(np.abs(heights[:-1] - heights[1:])))

    def get_holes(self, grid: np.ndarray) -> int:
        """
        Counts empty cells that have at least one filled cell above them.
        This is the most critical penalty for Tetris AI.
        """
        is_filled = grid != 0
        # A hole exists at (y, x) if grid[y, x] is empty AND there is a filled block somewhere in grid[0:y, x]
        # We can calculate this by taking the cumulative sum down the columns.
        # If the cumsum > 0 and the current cell is empty, it's a hole.
        filled_above = np.cumsum(is_filled, axis=0)
        holes = (filled_above > 0) & (~is_filled)
        return int(np.sum(holes))

    def calculate_reward(self, 
                         grid: np.ndarray, 
                         step_metrics: Dict[str, Any]) -> float:
        """Synthesizes the complete reward for a single step."""
        if step_metrics.get("game_over", False):
            return self.config.penalty_game_over

        reward = self.config.reward_survival
        
        # 1. Line Clear Rewards
        lines = step_metrics.get("lines_cleared", 0)
        if lines > 0:
            base_line_reward = lines * self.config.reward_lines_cleared
            if lines == 4:
                base_line_reward *= self.config.reward_tetris_multiplier
            reward += base_line_reward

        # 2. Board Topology Penalties (Only calculated when a piece locks to save CPU)
        if step_metrics.get("locked", False):
            heights = self.get_column_heights(grid)
            holes = self.get_holes(grid)
            bumpiness = self.get_bumpiness(heights)
            max_height = np.max(heights)

            reward += (holes * self.config.penalty_hole)
            reward += (bumpiness * self.config.penalty_bumpiness)
            reward += (max_height * self.config.penalty_height)

        return reward