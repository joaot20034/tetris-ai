import os
import pygame
from stable_baselines3 import DQN

from ai.config import RLConfig
from ai.agent import TetrisEnv

def evaluate_model(model_path: str, num_episodes: int = 5):
    """Loads a trained model and plays Tetris visibly without exploration."""
    
    # Explicitly start the Pygame engine to prevent video system errors
    pygame.init()
    
    # Check if the model exists (Stable-Baselines3 appends .zip automatically)
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Could not find trained model at '{model_path}.zip'")
        print("Make sure your AI has saved a checkpoint or finished training!")
        pygame.quit()
        return

    print(f"🧠 Loading trained brain from {model_path}...")
    
    # 1. Initialize environment in 'human' mode to enforce rendering
    config = RLConfig()
    env = TetrisEnv(config, render_mode="human")
    
    # 2. Load the trained DQN model
    model = DQN.load(model_path, env=env)
    
    # 3. Play the games
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        print(f"\n▶️ Starting Episode {episode}...")
        
        while not done:
            # Allow the user to safely click the 'X' on the window to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("🛑 Evaluation stopped by user.")
                    env.close()
                    pygame.quit()
                    return

            # CRITICAL: deterministic=True forces the AI to use 100% of its learned 
            # policy and 0% random exploration. It plays its absolute best.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            total_reward += reward
            done = terminated or truncated
            
            # Cap the framerate so it plays at a watchable human speed
            pygame.time.Clock().tick(60)

        # Print the final stats for the episode
        score = info.get('score', 0)
        lines = info.get('lines', 0)
        print(f"✅ Episode {episode} Finished | Score: {score} | Lines Cleared: {lines} | Total Reward: {total_reward:.1f}")

    print("\n🎉 Evaluation complete.")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    # We load the final model saved by train.py
    # If you want to load a specific checkpoint, change this string to match the file 
    # (e.g., "models/dqn_tetris_150000_steps")
    TARGET_MODEL = "models/dqn_tetris_final" 
    
    evaluate_model(TARGET_MODEL, num_episodes=5)