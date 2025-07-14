from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import sys
import os

# Since we're now inside the restaurant_env_updated folder, we can import directly
from restaurant_env import RestaurantEnv

# Create the RL environment directly instead of using gym.make
# Create a custom environment with longer episodes for better visualization
class LongEpisodeRestaurantEnv(RestaurantEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.max_episode_steps = 1000  # Longer episodes for better visualization

env = LongEpisodeRestaurantEnv(render_mode="human")

# Create only the neural network (RLModule) from our algorithm checkpoint.
# The checkpoint path structure for our restaurant environment
checkpoint_path = Path("D:/MachineLearning_Projects/new_venv/PPO_2025-07-13_13-31-31/PPO_RestaurantEnv-v0_b1285_00000_0_2025-07-13_13-31-33/checkpoint_000000")

rl_module = RLModule.from_checkpoint(
    str(checkpoint_path
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy")
)

# Run multiple episodes
num_episodes = 3
episode_returns = []

for episode in range(num_episodes):
    print(f"\n=== Starting Episode {episode + 1}/{num_episodes} ===")
    
    episode_return = 0.0
    done = False
    step_count = 0

    # Reset the env to get the initial observation.
    obs, info = env.reset()

    while not done:
        # Render the environment to see the pygame visualization
        env.render()

        # Convert observation dict to flattened tensor for the model
        # The model expects flattened observations based on the training configuration
        obs_tensor = torch.tensor([], dtype=torch.float32)
        
        # Flatten all observation components
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor = torch.cat([obs_tensor, torch.from_numpy(value).flatten().float()])
            else:
                obs_tensor = torch.cat([obs_tensor, torch.tensor([value], dtype=torch.float32)])
        
        # Add batch dimension
        obs_batch = obs_tensor.unsqueeze(0)
        
        # Compute the next action from a batch (B=1) of observations.
        model_outputs = rl_module.forward_inference({"obs": obs_batch})

        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

        # For discrete actions, take the argmax over the logits
        # The action space is a dict with 4 discrete components: type, waiter_id, customer_id, table_id
        action_type = np.argmax(action_dist_params[:4])  # First 4 values for action type
        waiter_id = np.argmax(action_dist_params[4:14])  # Next 10 values for waiter_id (0-9)
        customer_id = np.argmax(action_dist_params[14:64])  # Next 50 values for customer_id (0-49)
        table_id = np.argmax(action_dist_params[64:74])  # Last 10 values for table_id (0-9)
        
        # Create the action dictionary
        greedy_action = {
            'type': int(action_type),
            'waiter_id': int(waiter_id),
            'customer_id': int(customer_id),
            'table_id': int(table_id)
        }

        # Send the action to the environment for the next step.
        obs, reward, terminated, truncated, info = env.step(greedy_action)

        # Perform env-loop bookkeeping.
        episode_return += reward
        step_count += 1
        done = terminated or truncated
        
        # Add a small delay to make the visualization more visible
        import time
        time.sleep(0.1)  # 100ms delay between steps

    episode_returns.append(episode_return)
    print(f"Episode {episode + 1} completed with return: {episode_return:.2f}")
    print(f"Steps taken: {step_count}")
    print(f"Episode stats: {info}")

print(f"\n=== Summary ===")
print(f"Average episode return: {np.mean(episode_returns):.2f}")
print(f"Best episode return: {max(episode_returns):.2f}")
print(f"Worst episode return: {min(episode_returns):.2f}") 