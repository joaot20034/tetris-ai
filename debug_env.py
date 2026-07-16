import numpy as np
from stable_baselines3.common.env_checker import check_env
from ai.config import RLConfig
from ai.agent import TetrisEnv

def print_tensor_stats(name, tensor):
    tensor = np.array(tensor)
    print(f"  Shape:       {tensor.shape}")
    print(f"  Data Type:   {tensor.dtype}")
    print(f"  Min / Max:   {np.min(tensor)} / {np.max(tensor)}")
    print(f"  Unique Vals: {np.unique(tensor)}")

def run_diagnostics():
    print("🚀 Starting Phase 6: Environment Integrity Check...")
    config = RLConfig()
    
    # Initialize env with debug_mode ON
    env = TetrisEnv(config, debug_mode=True)
    
    try:
        check_env(env, warn=True)
        print("✅ Environment passes all Gymnasium API contracts.")
    except AssertionError as e:
        print(f"❌ Gymnasium API Violation: {e}")
        return

    print("\\n🚀 Starting Phase 3: State Verification...")
    obs, info = env.reset()
    
    print("Observation Dictionary Breakdown:")
    for key, tensor in obs.items():
        print(f"\\nKey: {key}")
        print_tensor_stats(key, tensor)

    print("\\n🚀 Starting Phase 1 & 2: Step Simulation...")
    print("Executing 5 manual steps to verify telemetry outputs...")
    
    # Simulate a sequence of arbitrary actions
    test_actions = [0, 1, 2, 2, 4] 
    for i, action in enumerate(test_actions):
        _obs, _reward, done, _truncated, _info = env.step(action)
        if done:
            print("Game Over triggered during simulation.")
            break

    print("\\n✅ Diagnostic simulation complete.")

if __name__ == "__main__":
    run_diagnostics()