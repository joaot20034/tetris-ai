import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ai.agent import TetrisEnv
from ai.config import RLConfig
from ai.network import TetrisFeatureExtractor

class TetrisTrainer:
    """Manages the lifecycle of the RL agent (Training, Saving, Loading)."""
    
    # ADDED: load_path parameter
    def __init__(self, config: RLConfig, render_mode: str = None, log_dir: str = "./logs", model_dir: str = "./models", load_path: str = None):
        self.config = config
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.env = Monitor(
            TetrisEnv(config, render_mode=render_mode), 
            filename=os.path.join(log_dir, "monitor.csv")
        )
        
        # NEW LOGIC: Check if we should resume from a saved model
        if load_path and os.path.exists(load_path + ".zip"):
            print(f"♻️ Resuming training from existing model: {load_path}.zip")
            self.model = DQN.load(load_path, env=self.env, tensorboard_log=self.log_dir)
        else:
            print("🆕 Starting a brand new model from scratch...")
            policy_kwargs = dict(
                features_extractor_class=TetrisFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512),
            )
            
            self.model = DQN(
                "MultiInputPolicy",
                self.env,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                gamma=config.gamma,
                tau=config.tau,
                target_update_interval=config.target_update_interval,
                exploration_initial_eps=config.exploration_initial_eps,
                exploration_final_eps=config.exploration_final_eps,
                exploration_fraction=config.exploration_fraction,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.log_dir,
                verbose=1
            )

    def train(self, total_timesteps: int = 1_000_000) -> None:
        """Starts the training loop and saves the final model."""
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Save a checkpoint every 50,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=self.model_dir,
            name_prefix="dqn_tetris"
        )
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            tb_log_name="DQN_Run",
            progress_bar=True
        )
        
        # Save the final model
        self.model.save(os.path.join(self.model_dir, "dqn_tetris_final"))
        print("Training complete and model saved.")

    def load_and_play(self, model_path: str, episodes: int = 5) -> None:
        """Loads a trained model and plays it with human rendering."""
        eval_env = TetrisEnv(self.config, render_mode="human")
        model = DQN.load(model_path, env=eval_env)
        
        for ep in range(episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Deterministic=True means we pick the best action, no exploration
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                eval_env.render()
                total_reward += reward
                done = terminated or truncated
                
            print(f"Episode {ep + 1} finished with Reward: {total_reward:.2f}")
            
        eval_env.close()