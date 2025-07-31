import os
import sys
import numpy as np
import torch
import pygame
from pathlib import Path

# Import the environment
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment.world_builder_env import WorldBuilderEnv
except ImportError:
    print("Could not import WorldBuilderEnv. Make sure your PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Path to checkpoint ---
# NOTE: Use raw string for Windows paths or forward slashes to avoid escape issues.
checkpoint_path = Path(os.path.abspath("trained_models/PPO_2025-07-31_06-48-19/PPO_WorldBuilderEnv-v0_57c5e_00000_0_2025-07-31_06-48-19/checkpoint_000000")) \
    / "learner_group" / "learner" / "rl_module" / "default_policy"

if not checkpoint_path.exists():
    print(f"Checkpoint path does not exist: {checkpoint_path}")
    print("Please check the path and try again.")
    sys.exit(1)

# Try importing RLModule and loading checkpoint
try:
    from ray.rllib.core.rl_module import RLModule
    rl_module = RLModule.from_checkpoint(checkpoint_path)
    print("Successfully loaded trained policy!")
except Exception as e:
    print(f"Failed to load RLModule from checkpoint: {e}")
    print("Make sure the checkpoint path is correct and the checkpoint exists.")
    sys.exit(1)

# Create the RL environment to test against (same as was used for training earlier).
env = WorldBuilderEnv(render_mode="human", flatten_obs=True)

episode_return = 0.0
done = False

# Reset the env to get the initial observation.
obs, info = env.reset()

print("Starting inference with trained PPO agent...")
print("Press 'q' to quit, any other key to continue")
print("Pygame window should appear now...")

while not done:
    # Render the current state
    env.render()
    
    # Add visual feedback for AI thinking
    print("🤖 AI is analyzing the city and making a decision...")
    
    # Compute the next action from a batch (B=1) of observations.
    obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
    model_outputs = rl_module.forward_inference({"obs": obs_batch})

    # Extract the action distribution parameters from the output and dissolve batch dim.
    action_dist_params = model_outputs["action_dist_inputs"][0].detach().cpu().numpy()

    # For discrete actions (World Builder has 5 discrete actions), take the argmax over the logits:
    greedy_action = np.argmax(action_dist_params)
    
    # Add action name for better visualization
    action_names = ["Pass", "Build Farm", "Build Lumberyard", "Build Quarry", "Build House"]
    action_name = action_names[greedy_action]
    print(f"🎯 AI decided to: {action_name}")

    # Send the action to the environment for the next step.
    obs, reward, terminated, truncated, info = env.step(greedy_action)

    # Perform env-loop bookkeeping.
    episode_return += reward
    done = terminated or truncated

    # Print enhanced step info with emojis and better formatting
    print(f"🎮 Step Results:")
    print(f"   Action: {action_name}")
    print(f"   Reward: {reward:.2f} | Total: {episode_return:.2f}")
    
    # Defensive: check obs length before indexing
    if len(obs) >= 105:
        print(f"   📦 Resources: Food={obs[100]:.1f} | Wood={obs[101]:.1f} | Stone={obs[102]:.1f}")
        print(f"   👥 Population: {obs[103]:.1f}/{obs[104]:.1f}")
    else:
        print(f"   📊 Observation: {obs}")
    print("=" * 60)
    
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                done = True
    
    # Enhanced delay with visual feedback
    pygame.time.wait(1500)  # 1.5 second delay for better visualization

print(f"Reached episode return of {episode_return}.")
env.close()
