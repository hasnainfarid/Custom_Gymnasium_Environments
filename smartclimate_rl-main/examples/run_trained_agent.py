from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import smartclimate  # Ensure env is registered
import sys

# Usage: python run_trained_agent.py <checkpoint_dir>
if len(sys.argv) < 2:
    print("Usage: python run_trained_agent.py <checkpoint_dir>")
    sys.exit(1)

checkpoint_dir = Path(sys.argv[1])

# Load RLModule from checkpoint
rl_module = RLModule.from_checkpoint(
    checkpoint_dir / "learner_group" / "learner" / "rl_module" / "default_policy"
)

# Create the SmartClimate RL environment
env = gym.make("SmartClimateEnv-v0")

obs, info = env.reset()
done = False
episode_return = 0.0

while not done:
    # Convert obs to torch tensor with batch dim
    obs_batch = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    model_outputs = rl_module.forward_inference({"obs": obs_batch})
    # For Dict action space, RLlib returns a dict of action_dist_inputs
    action_dist_params = model_outputs["action_dist_inputs"]
    # For continuous ac_temp, take mean; for lights (MultiBinary), use sigmoid+0.5>0.5
    ac_temp_mean = action_dist_params["ac_temp"][0].detach().numpy()
    lights_logits = action_dist_params["lights"][0].detach().numpy()
    # ac_temp is shape (1,), clip to [16,32]
    ac_temp = np.clip(ac_temp_mean, env.action_space["ac_temp"].low, env.action_space["ac_temp"].high)
    # lights: convert logits to binary actions
    lights = (torch.sigmoid(torch.from_numpy(lights_logits)) > 0.5).int().numpy()
    action = {"ac_temp": ac_temp, "lights": lights}
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    episode_return += reward
    done = terminated or truncated

print(f"Reached episode return of {episode_return}.")
env.close() 