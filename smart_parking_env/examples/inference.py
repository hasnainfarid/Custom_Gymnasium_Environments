"""
Simple inference script for Smart Parking Environment with pygame visualization.
"""

from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import sys
import os
import time

# Import environment
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.parking_env import SmartParkingEnv
from utils.visualization import ParkingVisualization

def run_inference(checkpoint_path: str):
    """Run inference with trained policy and pygame visualization."""
    print("Smart Parking Environment - Policy Inference with Visualization")
    print("=" * 60)
    
    # Load the trained policy
    print(f"Loading policy from: {checkpoint_path}")
    rl_module = RLModule.from_checkpoint(
        Path(checkpoint_path)
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )
    print("✅ Policy loaded successfully!")
    
    # Create environment
    env = SmartParkingEnv()
    obs, info = env.reset()
    
    # Get actual components from environment
    parking_lot = env.parking_lot
    pricing_manager = env.pricing_manager
    
    print("🎮 Starting episode with pygame visualization...")
    print("Controls: SPACE=pause, 1-4=speed, ESC=quit")
    print("Tip: Use speed 4 (5x) for faster simulation!")
    print()
    
    # Run one episode with visualization using actual environment data
    episode_return = 0.0
    done = False
    timestep = 0
    
    # Use the visualization context manager like in test script
    with ParkingVisualization(parking_lot, pricing_manager) as viz:
        while not done:
            # Get action from policy
            obs_batch = torch.from_numpy(obs).unsqueeze(0).float()
            with torch.no_grad():
                model_outputs = rl_module.forward_inference({"obs": obs_batch})
            
            action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
            action = np.argmax(action_dist_params)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
            timestep += 1
            
            # Get actual customers from environment (like in test script)
            customers = list(env.parking_lot.queue)
            customer_stats = env.customer_manager.get_statistics()
            
            # Run pygame visualization with actual data (like in test script)
            viz_start_time = time.time()
            while time.time() - viz_start_time < 0.05:  # Show each step for 0.05 seconds
                if not viz.run_visualization(info['hour'], info['minute'], customers, 
                                           customer_stats, reward):
                    print("Visualization stopped by user.")
                    done = True
                    break
            
            # Print progress every 100 timesteps
            if timestep % 100 == 0:
                current_hour = timestep // 60
                current_minute = timestep % 60
                print(f"⏰ {current_hour:02d}:{current_minute:02d} | Return={episode_return:.2f} | Revenue=${info.get('total_revenue', 0):.2f}")
    
    # Print final results
    print(f"\n🎯 Episode Complete!")
    print(f"  Total Return: {episode_return:.2f}")
    print(f"  Total Revenue: ${info.get('total_revenue', 0):.2f}")
    print(f"  Rejections: {info.get('rejections', 0)}")
    print(f"  Occupancy Rate: {info.get('occupancy_rate', 0):.2%}")
    
    env.close()
    print("✅ Inference with visualization completed!")

if __name__ == "__main__":
    # Use your existing policy folder - correct path to checkpoint
    checkpoint_path = r"D:\MachineLearning_Projects\new_venv\smart_parking_env\Trained_PPO_Policy\PPO_SmartParkingEnv-v0_88e23_00000_0_2025-07-23_01-08-25\checkpoint_000000"
    run_inference(checkpoint_path) 