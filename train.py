from ai.config import RLConfig
from ai.trainer import TetrisTrainer
from ui.dashboard import UIDashboardCallback

if __name__ == "__main__":
    config = RLConfig()
    
    # We must set render_mode="human" for the UI to draw
    trainer = TetrisTrainer(config=config)
    trainer.env.envs[0].unwrapped.render_mode = "human"
    trainer.env.envs[0].unwrapped.renderer = trainer.env.envs[0].unwrapped.Renderer()
    
    # Initialize the dashboard callback
    dashboard = UIDashboardCallback()
    
    print("Starting Tetris AI Training...")
    print("Watch the Pygame window to see the agent learn.")
    
    try:
        # Pass the callback into the learn function
        trainer.model.learn(
            total_timesteps=2_000_000,
            callback=[dashboard],
            tb_log_name="DQN_Run"
        )
    except KeyboardInterrupt:
        print("\nTraining gracefully stopped by user.")
    finally:
        trainer.model.save("models/dqn_tetris_final")