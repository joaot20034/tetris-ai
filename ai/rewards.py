import numpy as np
from typing import Dict, Any, Tuple
from ai.config import RLConfig

class RewardShaper:
    """Calculates delta-based heuristic penalties and rewards for the RL agent."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # THE SURVIVAL FIX: Force survival reward to 0.0 to prevent infinite stalling
        self.config.reward_survival = 0.0

    # --- YOUR OPTIMIZED NUMPY MATH ---
    def get_column_heights(self, grid: np.ndarray) -> np.ndarray:
        is_filled = grid != 0
        heights = np.zeros(grid.shape[1], dtype=np.int32)
        for col in range(grid.shape[1]):
            filled_indices = np.where(is_filled[:, col])[0]
            if len(filled_indices) > 0:
                heights[col] = grid.shape[0] - filled_indices[0]
        return heights

    def get_bumpiness(self, heights: np.ndarray) -> int:
        return int(np.sum(np.abs(heights[:-1] - heights[1:])))

    def get_holes(self, grid: np.ndarray) -> int:
        is_filled = grid != 0
        filled_above = np.cumsum(is_filled, axis=0)
        holes = (filled_above > 0) & (~is_filled)
        return int(np.sum(holes))

    def get_board_metrics(self, grid: np.ndarray) -> dict:
        """Helper to grab all metrics at once."""
        heights = self.get_column_heights(grid)
        return {
            "holes": self.get_holes(grid),
            "bumpiness": self.get_bumpiness(heights),
            "height": int(np.max(heights))
        }

    # --- THE DELTA REWARD LOGIC ---
    def calculate_reward(self, 
                         prev_grid: np.ndarray, 
                         curr_grid: np.ndarray, 
                         step_metrics: Dict[str, Any]) -> Tuple[float, dict]:
        """Calculates the delta-based reward and returns a detailed breakdown dictionary."""
        
        breakdown = {}
        total_reward = 0.0
        
        game_over = step_metrics.get("game_over", False)
        lines = step_metrics.get("lines_cleared", 0)
        locked = step_metrics.get("locked", False)

        # 1. Terminal States
        if game_over:
            breakdown['Game Over'] = self.config.penalty_game_over
            return self.config.penalty_game_over, breakdown
        else:
            breakdown['Survival'] = self.config.reward_survival
            total_reward += self.config.reward_survival

        # 2. Line Clear Rewards
        if lines > 0:
            base_line_reward = lines * self.config.reward_lines_cleared
            if lines == 4:
                base_line_reward *= self.config.reward_tetris_multiplier
            breakdown[f'{lines} Line Clear'] = base_line_reward
            total_reward += base_line_reward
        else:
            breakdown['Line Clear'] = 0.0

        # 3. Delta Board Topology Penalties (Only checked when a piece locks)
        if locked:
            prev_metrics = self.get_board_metrics(prev_grid)
            curr_metrics = self.get_board_metrics(curr_grid)

            # Only penalize if the state got WORSE
            delta_holes = curr_metrics['holes'] - prev_metrics['holes']
            if delta_holes > 0:
                hole_reward = delta_holes * self.config.penalty_hole
                breakdown['New Holes Created'] = hole_reward
                total_reward += hole_reward
            else:
                breakdown['New Holes Created'] = 0.0

            delta_bump = curr_metrics['bumpiness'] - prev_metrics['bumpiness']
            if delta_bump > 0:
                bump_reward = delta_bump * self.config.penalty_bumpiness
                breakdown['Increased Bumpiness'] = bump_reward
                total_reward += bump_reward
            else:
                breakdown['Increased Bumpiness'] = 0.0

            delta_height = curr_metrics['height'] - prev_metrics['height']
            if delta_height > 0:
                height_reward = delta_height * self.config.penalty_height
                breakdown['Increased Height'] = height_reward
                total_reward += height_reward
            else:
                breakdown['Increased Height'] = 0.0
        else:
            breakdown['New Holes Created'] = 0.0
            breakdown['Increased Bumpiness'] = 0.0
            breakdown['Increased Height'] = 0.0

        return total_reward, breakdown