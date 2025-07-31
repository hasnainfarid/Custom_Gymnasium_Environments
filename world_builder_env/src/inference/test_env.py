#!/usr/bin/env python3
"""
Simple test script for the World Builder Game environment with pygame rendering.
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.world_builder_env import WorldBuilderEnv

def test_environment_with_render():
    """Test the environment functionality with pygame rendering."""
    print("Testing World Builder Environment (with pygame render)...")
    
    # Create environment with rendering enabled
    env = WorldBuilderEnv(render_mode="human")
    
    # Test reset
    obs, info = env.reset()
    print("✓ Environment reset successful")
    print(f"  Initial resources: {info['resources']}")
    print(f"  Initial population: {info['population']}/{info['population_capacity']}")
    
    # Test action space
    print(f"✓ Action space: {env.action_space}")
    print(f"✓ Observation space: {env.observation_space}")
    
    # Test a few steps with rendering
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the environment
        env.render()
        time.sleep(0.4)  # Slow down to see the rendering
        
        print(f"  Step {step + 1}: Action={action}, Reward={reward:.1f}, Done={done}")
        print(f"    Resources: {info['resources']}")
        print(f"    Population: {info['population']}/{info['population_capacity']}")
        
        if done:
            print(f"    Episode ended after {step + 1} steps")
            break
    
    print(f"✓ Total reward: {total_reward:.1f}")
    
    # Test multiple episodes with rendering
    print("\nTesting multiple episodes (with rendering)...")
    num_episodes = 100  # Changed to 50 episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 20:  # Limit steps per episode
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render the environment
            env.render()
            time.sleep(0.2)
            
            if done:
                break
        
        print(f"  Episode {episode + 1}: {steps} steps, reward {episode_reward:.1f}")
    
    env.close()
    print("✓ All tests passed with rendering!")


if __name__ == "__main__":
    test_environment_with_render()
    print("\n🎉 Environment test with pygame rendering completed successfully!") 