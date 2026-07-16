import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from game.game import TetrisGame, Action
from game.board import PIECE_IDS
from game.renderer import Renderer
from ai.rewards import RewardShaper
from ai.config import RLConfig

class TetrisEnv(gym.Env):
    """
    Standard Gymnasium Environment for Tetris.
    Compatible with Stable-Baselines3.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: RLConfig, render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        self.game = TetrisGame()
        self.reward_shaper = RewardShaper(config)
        self.renderer = Renderer() if render_mode == "human" else None
        
        # Action Space: 7 discrete actions (NOOP, LEFT, RIGHT, SOFT_DROP, HARD_DROP, ROTATE_CW, HOLD)
        # Note: We omit ROTATE_CCW to simplify the action space; CW is mathematically sufficient.
        self.action_space = spaces.Discrete(7)
        
        # Observation Space: A dictionary allowing MultiInputPolicy to route data correctly
        self.observation_space = spaces.Dict({
            # The 20x10 board. We cap the ID at 7 (the max piece ID).
            "board": spaces.Box(low=0, high=7, shape=(20, 10), dtype=np.int8),
            
            # Current falling piece ID (0 if none, 1-7 for pieces)
            "current_piece": spaces.Discrete(8),
            
            # Next 5 pieces in the queue
            "next_pieces": spaces.MultiDiscrete([8] * 5),
            
            # Currently held piece (0 if none)
            "hold_piece": spaces.Discrete(8),
            
            # Boolean flag indicating if we can currently swap with the hold piece
            "can_hold": spaces.Discrete(2)
        })

        self.step_count = 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Translates the internal TetrisGame state into the Gymnasium ObservationSpace."""
        # Get piece IDs safely
        current_id = PIECE_IDS.get(self.game.current_piece.name, 0)
        hold_id = PIECE_IDS.get(self.game.hold_piece, 0) if self.game.hold_piece else 0
        
        # Map the next 5 queue items to their integer IDs
        queue_names = self.game.bag.get_queue(5)
        next_ids = np.array([PIECE_IDS.get(name, 0) for name in queue_names], dtype=np.int64)

        return {
            "board": np.copy(self.game.board.grid),
            "current_piece": current_id,
            "next_pieces": next_ids,
            "hold_piece": hold_id,
            "can_hold": 1 if self.game.can_hold else 0
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment to start a new episode."""
        super().reset(seed=seed)
        self.game.reset()
        self.step_count = 0
        
        # info dict can contain telemetry for TensorBoard
        info = {
            "score": self.game.score,
            "lines": self.game.lines_cleared_total
        }
        
        return self._get_obs(), info

    def step(self, action_idx: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        self.step_count += 1
        
        # Map integer from SB3 (0-6) to our game Action IntEnum
        # Note: If Action Enum has ROTATE_CCW, we skip it by manipulating the index if necessary,
        # but here we can just use mapping.
        action_mapping = [
            Action.NOOP, Action.LEFT, Action.RIGHT, 
            Action.SOFT_DROP, Action.HARD_DROP, Action.ROTATE_CW, Action.HOLD
        ]
        game_action = action_mapping[action_idx]
        
        # 1. Step the game state
        step_metrics = self.game.step(game_action)
        
        # 2. Automatically apply gravity every few steps if the agent isn't dropping
        # This forces the agent to deal with the falling piece rather than stalling
        if self.step_count % 10 == 0 and not step_metrics.get("locked"):
            gravity_metrics = self.game.apply_gravity()
            # Merge metrics (if gravity locked the piece, override the locked flag)
            if gravity_metrics.get("locked"):
                step_metrics = gravity_metrics

        # 3. Calculate Shaped Reward
        reward = self.reward_shaper.calculate_reward(self.game.board.grid, step_metrics)

        # 4. Check Termination (Game Over) or Truncation (Max Steps reached)
        terminated = step_metrics.get("game_over", False)
        truncated = self.step_count >= self.config.max_steps_per_episode

        # 5. Compile info for logging
        info = {
            "score": self.game.score,
            "lines": self.game.lines_cleared_total,
            "level": self.game.level
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        """Renders the environment to the screen."""
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self.game)

    def close(self) -> None:
        """Cleans up resources."""
        if self.renderer:
            self.renderer.close()