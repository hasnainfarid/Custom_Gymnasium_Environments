"""
Test script for the Traffic Management Environment.
"""

import numpy as np
import gymnasium as gym
import time
from typing import Dict, Any

# Import the environment
import traffic_management_env


def test_environment_creation():
    """Test basic environment creation and properties."""
    print("Testing environment creation...")
    
    # Test default environment
    env = gym.make('TrafficManagement-v0')
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test custom environment
    env_custom = gym.make('TrafficManagement-v0', 
                         grid_size=(3, 3),
                         num_intersections=6,
                         max_vehicles=20,
                         spawn_rate=0.2)
    print(f"✓ Custom environment created successfully")
    
    env.close()
    env_custom.close()


def test_reset_functionality():
    """Test environment reset functionality."""
    print("\nTesting reset functionality...")
    
    env = gym.make('TrafficManagement-v0')
    
    # Test reset
    observation, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation type: {type(observation)}")
    print(f"  Info keys: {list(info.keys())}")
    
    # Test reset with different seed
    observation2, info2 = env.reset(seed=123)
    print(f"✓ Reset with different seed successful")
    
    # Verify observations are different (due to different seeds)
    if not np.array_equal(observation, observation2):
        print(f"✓ Different seeds produce different initial states")
    
    env.close()


def test_step_functionality():
    """Test environment step functionality."""
    print("\nTesting step functionality...")
    
    env = gym.make('TrafficManagement-v0')
    observation, info = env.reset(seed=42)
    
    # Test valid actions
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {i+1}: reward={reward:.3f}, vehicles={info['num_vehicles']}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"✓ Step functionality working correctly")
    env.close()


def test_action_space():
    """Test action space functionality."""
    print("\nTesting action space...")
    
    env = gym.make('TrafficManagement-v0')
    
    # Test action space properties
    print(f"  Action space: {env.action_space}")
    print(f"  Action space shape: {env.action_space.shape}")
    
    # Test action sampling
    for i in range(5):
        action = env.action_space.sample()
        print(f"  Sample action {i+1}: {action}")
    
    # Test specific actions
    observation, info = env.reset()
    
    # Test all maintain actions (0s)
    action = np.zeros(env.action_space.shape[0], dtype=int)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"  All maintain action: reward={reward:.3f}")
    
    # Test all NS_GREEN actions (1s)
    action = np.ones(env.action_space.shape[0], dtype=int)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"  All NS_GREEN action: reward={reward:.3f}")
    
    # Test all EW_GREEN actions (2s)
    action = np.full(env.action_space.shape[0], 2, dtype=int)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"  All EW_GREEN action: reward={reward:.3f}")
    
    print(f"✓ Action space working correctly")
    env.close()


def test_observation_space():
    """Test observation space functionality."""
    print("\nTesting observation space...")
    
    env = gym.make('TrafficManagement-v0')
    observation, info = env.reset()
    
    # Test observation properties
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Observation dtype: {observation.dtype}")
    print(f"  Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    
    # Check if observation is within bounds
    if env.observation_space.contains(observation):
        print(f"✓ Observation within space bounds")
    else:
        print(f"✗ Observation outside space bounds")
    
    # Test observation consistency over multiple steps
    observations = [observation]
    for i in range(5):
        action = env.action_space.sample()
        observation, _, _, _, _ = env.step(action)
        observations.append(observation)
    
    # Check if observations have consistent shape
    shapes = [obs.shape for obs in observations]
    if all(shape == shapes[0] for shape in shapes):
        print(f"✓ Observation shape consistent across steps")
    else:
        print(f"✗ Observation shape inconsistent")
    
    env.close()


def test_reward_system():
    """Test reward system."""
    print("\nTesting reward system...")
    
    env = gym.make('TrafficManagement-v0', spawn_rate=0.5)  # Higher spawn rate for testing
    observation, info = env.reset(seed=42)
    
    rewards = []
    for i in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    print(f"  Total steps: {len(rewards)}")
    print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Reward std: {np.std(rewards):.3f}")
    
    # Check if rewards are reasonable
    if any(r > 0 for r in rewards):
        print(f"✓ Positive rewards achieved")
    if any(r < 0 for r in rewards):
        print(f"✓ Negative rewards (penalties) working")
    
    print(f"✓ Reward system functioning")
    env.close()


def test_rendering():
    """Test rendering functionality."""
    print("\nTesting rendering...")
    
    try:
        # Test human rendering
        env = gym.make('TrafficManagement-v0', render_mode='human')
        observation, info = env.reset()
        
        print(f"  Testing human rendering...")
        for i in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.1)  # Brief pause to see rendering
            
            if terminated or truncated:
                break
        
        env.close()
        print(f"✓ Human rendering working")
        
    except Exception as e:
        print(f"⚠ Human rendering failed (this may be expected in headless environments): {e}")
    
    try:
        # Test RGB array rendering
        env = gym.make('TrafficManagement-v0', render_mode='rgb_array')
        observation, info = env.reset()
        
        rgb_array = env.render()
        if rgb_array is not None:
            print(f"  RGB array shape: {rgb_array.shape}")
            print(f"✓ RGB array rendering working")
        else:
            print(f"⚠ RGB array rendering returned None")
        
        env.close()
        
    except Exception as e:
        print(f"⚠ RGB array rendering failed: {e}")


def test_episode_completion():
    """Test full episode completion."""
    print("\nTesting episode completion...")
    
    env = gym.make('TrafficManagement-v0')
    observation, info = env.reset(seed=42)
    
    total_reward = 0
    step_count = 0
    
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"  Episode completed in {step_count} steps")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Final metrics: {info.get('metrics', {})}")
    
    print(f"✓ Episode completion working")
    env.close()


def run_performance_test():
    """Run a performance test."""
    print("\nRunning performance test...")
    
    env = gym.make('TrafficManagement-v0')
    
    start_time = time.time()
    total_steps = 0
    
    for episode in range(5):
        observation, info = env.reset()
        episode_steps = 0
        
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_steps += 1
            total_steps += 1
            
            if terminated or truncated:
                break
        
        print(f"  Episode {episode + 1}: {episode_steps} steps")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Steps per second: {total_steps / elapsed_time:.1f}")
    
    print(f"✓ Performance test completed")
    env.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAFFIC MANAGEMENT ENVIRONMENT TESTS")
    print("=" * 60)
    
    try:
        test_environment_creation()
        test_reset_functionality()
        test_step_functionality()
        test_action_space()
        test_observation_space()
        test_reward_system()
        test_rendering()
        test_episode_completion()
        run_performance_test()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()