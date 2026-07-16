from dataclasses import dataclass

@dataclass
class RLConfig:
    """Hyperparameters for Reinforcement Learning training."""
    
    # Environment Settings
    max_steps_per_episode: int = 100_000
    
    # DQN Hyperparameters (Ready to be swapped to PPO later if needed)
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99           # Discount factor
    tau: float = 0.05             # Soft update coefficient
    target_update_interval: int = 10_000
    
    # Exploration (Epsilon-Greedy)
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.2  # Fraction of training time to decay epsilon over
    
    # Reward Shaping Weights
    reward_lines_cleared: float = 10.0
    reward_tetris_multiplier: float = 2.0  # Bonus for clearing 4 lines at once
    penalty_hole: float = -0.5
    penalty_bumpiness: float = -0.2
    penalty_height: float = -0.1
    penalty_game_over: float = -50.0
    reward_survival: float = 0.01  # Small reward for staying alive