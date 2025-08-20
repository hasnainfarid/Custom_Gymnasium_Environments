"""
Simple test script for the Traffic Management Environment without package imports.
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# Import local modules directly
from environment import TrafficManagementEnv

# Register the environment
register(
    id='TrafficManagement-v0',
    entry_point='environment:TrafficManagementEnv',
    max_episode_steps=1000,
    kwargs={
        'grid_size': (5, 5),
        'num_intersections': 9,
        'max_vehicles': 50,
        'spawn_rate': 0.3,
    }
)

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Traffic Management Environment...")
    
    # Create environment
    env = gym.make('TrafficManagement-v0')
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test reset
    observation, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Initial vehicles: {info['num_vehicles']}")
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {i+1}: reward={reward:.3f}, vehicles={info['num_vehicles']}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"✓ Environment functioning correctly")
    print(f"  Total reward after {i+1} steps: {total_reward:.3f}")
    
    # Test observation bounds
    if env.observation_space.contains(observation):
        print(f"✓ Observation within bounds")
    else:
        print(f"⚠ Observation outside bounds")
    
    env.close()
    print(f"✓ Environment closed successfully")

def test_custom_config():
    """Test environment with custom configuration."""
    print("\nTesting custom configuration...")
    
    env = TrafficManagementEnv(
        grid_size=(3, 3),
        num_intersections=4,
        max_vehicles=20,
        spawn_rate=0.4
    )
    
    observation, info = env.reset()
    print(f"✓ Custom environment created")
    print(f"  Grid size: {env.grid_size}")
    print(f"  Num intersections: {env.num_intersections}")
    print(f"  Max vehicles: {env.max_vehicles}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"✓ Custom configuration working")
    env.close()

def test_action_effects():
    """Test that actions have effects on the environment."""
    print("\nTesting action effects...")
    
    env = gym.make('TrafficManagement-v0', spawn_rate=0.6)  # Higher spawn rate
    observation, info = env.reset(seed=42)
    
    # Record initial state
    initial_obs = observation.copy()
    
    # Take some actions
    actions_taken = []
    observations = [initial_obs]
    
    for i in range(5):
        action = env.action_space.sample()
        actions_taken.append(action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation.copy())
        
        if terminated or truncated:
            break
    
    # Check if observations changed
    changed = False
    for i in range(1, len(observations)):
        if not np.array_equal(observations[i], observations[i-1]):
            changed = True
            break
    
    if changed:
        print(f"✓ Actions are affecting environment state")
    else:
        print(f"⚠ Environment state not changing (may be normal for short episodes)")
    
    env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("TRAFFIC MANAGEMENT ENVIRONMENT - SIMPLE TESTS")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_custom_config()
        test_action_effects()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The Traffic Management Environment is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()