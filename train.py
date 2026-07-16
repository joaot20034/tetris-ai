from ai.config import RLConfig
from ai.trainer import TetrisTrainer
from ui.dashboard import UIDashboardCallback

if __name__ == "__main__":
    config = RLConfig()
    
    # Target our final save file
    SAVE_PATH = "models/dqn_tetris_final"
    
    # Pass the load_path to the trainer
    trainer = TetrisTrainer(config=config, render_mode="human", load_path=SAVE_PATH)
    
    dashboard = UIDashboardCallback()
    
    print("Starting Tetris AI Training Dashboard...")
    print("Press [SPACE] to pause. Use [UP/DOWN] arrows to adjust speed up to 1000x.")
    
    try:
        trainer.model.learn(
            total_timesteps=2_000_000,
            callback=[dashboard],
            tb_log_name="DQN_Run",
            reset_num_timesteps=False  # IMPORTANT: Keeps TensorBoard charts continuous
        )
    except KeyboardInterrupt:
        print("\nTraining gracefully suspended by user.")
    finally:
        trainer.model.save(SAVE_PATH)
        print(f"Progress saved to {SAVE_PATH}.zip")