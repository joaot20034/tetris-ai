import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hashlib
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

    def __init__(self, config: RLConfig, render_mode: Optional[str] = None, debug_mode: bool = False):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        self.debug_mode = debug_mode
        
        self.game = TetrisGame()
        self.reward_shaper = RewardShaper(config)
        self.renderer = Renderer() if render_mode == "human" else None
        
        # Action Space: 7 discrete actions
        self.action_space = spaces.Discrete(7)
        
        # Normalized Observation Space for CNN compatibility
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0.0, high=1.0, shape=(20, 10), dtype=np.float32),
            "current_piece": spaces.Discrete(8),
            "next_pieces": spaces.MultiDiscrete([8] * 5),
            "hold_piece": spaces.Discrete(8),
            "can_hold": spaces.Discrete(2)
        })

        self.step_count = 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Translates the internal TetrisGame state into the Gymnasium ObservationSpace."""
        current_id = PIECE_IDS.get(self.game.current_piece.name, 0) if self.game.current_piece else 0
        hold_id = PIECE_IDS.get(self.game.hold_piece, 0) if self.game.hold_piece else 0
        
        queue_names = self.game.bag.get_queue(5)
        next_ids = np.array([PIECE_IDS.get(name, 0) for name in queue_names], dtype=np.int64)

        # 1. Grab the static locked board
        raw_board = np.array(self.game.board.grid, dtype=np.float32)
        
        # 2. BLIND AI FIX: Draw the falling piece onto the observation array
        cp = self.game.current_piece
        if cp and hasattr(cp, 'shape'):
            for row in range(len(cp.shape)):
                for col in range(len(cp.shape[row])):
                    if cp.shape[row][col] != 0:
                        board_y = cp.y + row
                        board_x = cp.x + col
                        # Ensure we don't draw outside the array boundaries
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            raw_board[board_y][board_x] = 1.0

        # 3. Normalize the final combined image
        norm_board = np.where(raw_board > 0, 1.0, 0.0).astype(np.float32)

        return {
            "board": norm_board,
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
        
        info = {
            "score": self.game.score,
            "lines": self.game.lines_cleared_total
        }
        
        return self._get_obs(), info

    def step(self, action_idx: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        self.step_count += 1
        
        # Capture the board state BEFORE the action for Delta rewards
        prev_grid = np.copy(self.game.board.grid)

        # Safely capture pre-action coordinates to detect useless moves
        cp = self.game.current_piece
        prev_x = getattr(cp, 'x', 0) if cp else 0
        prev_y = getattr(cp, 'y', 0) if cp else 0
        prev_rot = getattr(cp, 'rotation', 0) if cp else 0

        # Map integer from SB3 (0-6) to our game Action IntEnum
        action_mapping = [
            Action.NOOP, Action.LEFT, Action.RIGHT, 
            Action.SOFT_DROP, Action.HARD_DROP, Action.ROTATE_CW, Action.HOLD
        ]
        game_action = action_mapping[action_idx]
        
        # 1. Step the game state
        step_metrics = self.game.step(game_action)
        
        # Capture post-action coordinates
        cp = self.game.current_piece
        curr_x = getattr(cp, 'x', 0) if cp else 0
        curr_y = getattr(cp, 'y', 0) if cp else 0
        curr_rot = getattr(cp, 'rotation', 0) if cp else 0

        # Did the agent actually change the piece's physical state?
        state_changed = (prev_x != curr_x) or (prev_y != curr_y) or (prev_rot != curr_rot)

        # 2. THE GRAVITY ENFORCER
        # If the action did absolutely nothing (hit a wall, NOOP, blocked rotation), force gravity.
        if not state_changed and not step_metrics.get("locked"):
            gravity_metrics = self.game.apply_gravity()
            if gravity_metrics.get("locked"):
                step_metrics = gravity_metrics
                
        # Apply natural gravity every 10 steps to ensure downward momentum
        elif self.step_count % 10 == 0 and not step_metrics.get("locked"):
            gravity_metrics = self.game.apply_gravity()
            if gravity_metrics.get("locked"):
                step_metrics = gravity_metrics

        # Capture the board state AFTER the action for Delta rewards
        curr_grid = np.copy(self.game.board.grid)

        # 3. Calculate Delta-Shaped Reward
        reward, reward_breakdown = self.reward_shaper.calculate_reward(
            prev_grid, curr_grid, step_metrics
        )

        # 4. Check Termination or Truncation
        terminated = step_metrics.get("game_over", False)
        truncated = self.step_count >= self.config.max_steps_per_episode

        # 5. Compile info for logging
        info = {
            "score": self.game.score,
            "lines": self.game.lines_cleared_total,
            "level": self.game.level,
            "reward_breakdown": reward_breakdown
        }

        # 6. Extract observations
        obs = self._get_obs()

        # Phase 1 & 2: Debug Telemetry Execution
        if self.debug_mode:
            self._debug_log_step(game_action.name, reward, info, prev_grid, curr_grid, obs)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Renders the environment to the screen."""
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self.game)

    def close(self) -> None:
        """Cleans up resources."""
        if self.renderer:
            self.renderer.close()

    def _debug_log_step(self, action_name: str, reward: float, info: dict, prev_grid: np.ndarray, curr_grid: np.ndarray, obs: dict):
        """Phase 1 & 2: Console formatting for forensic analysis."""
        print(f"\n{'='*35}")
        print(f"STEP {self.step_count} | ACTION: {action_name}")
        print("--- Reward Breakdown ---")
        
        # Safely print each calculated reward component
        for key, value in info.get('reward_breakdown', {}).items():
            print(f"{value:>+7.2f}  {key}")
        
        print("-" * 24)
        print(f"TOTAL  {reward:>+7.2f}")
        
        print("\n--- Environment Deltas ---")
        # Recalculate metrics strictly for printing the visual difference
        prev_m = self.reward_shaper.get_board_metrics(prev_grid)
        curr_m = self.reward_shaper.get_board_metrics(curr_grid)
        
        print(f"Holes:     {prev_m['holes']} -> {curr_m['holes']}")
        print(f"Height:    {prev_m['height']} -> {curr_m['height']}")
        print(f"Bumpiness: {prev_m['bumpiness']} -> {curr_m['bumpiness']}")
        
        # Hash detects if an action did absolutely nothing (infinite loop vulnerability)
        board_hash = hashlib.md5(np.ascontiguousarray(obs['board'])).hexdigest()
        print(f"Board Hash: {board_hash[:8]}")
        print(f"{'='*35}")