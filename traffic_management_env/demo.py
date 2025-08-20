"""
Demonstration script for the Traffic Management Environment.

This script shows how to use the environment for training and evaluation.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import time

# Import the environment
from environment import TrafficManagementEnv

# Register the environment
register(
    id='TrafficManagement-v0',
    entry_point='environment:TrafficManagementEnv',
    max_episode_steps=1000,
    kwargs={
        'grid_size': (4, 4),
        'num_intersections': 8,
        'max_vehicles': 40,
        'spawn_rate': 0.4,
    }
)

def random_policy_demo():
    """Demonstrate the environment with a random policy."""
    print("=" * 60)
    print("RANDOM POLICY DEMONSTRATION")
    print("=" * 60)
    
    env = gym.make('TrafficManagement-v0')
    
    # Run multiple episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        step_count = 0
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print periodic updates
            if step_count % 100 == 0:
                metrics = info.get('metrics', {})
                print(f"  Step {step_count}: "
                      f"Reward={total_reward:.2f}, "
                      f"Vehicles={info['num_vehicles']}, "
                      f"Avg Wait={metrics.get('average_waiting_time', 0):.2f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        final_metrics = info.get('metrics', {})
        print(f"  Final: Steps={step_count}, "
              f"Total Reward={total_reward:.2f}, "
              f"Vehicles Passed={final_metrics.get('total_vehicles_passed', 0)}, "
              f"Avg Queue Length={final_metrics.get('average_queue_length', 0):.2f}")
    
    env.close()
    
    print(f"\nSummary:")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


def smart_policy_demo():
    """Demonstrate a simple heuristic policy."""
    print("\n" + "=" * 60)
    print("SMART HEURISTIC POLICY DEMONSTRATION")
    print("=" * 60)
    
    env = gym.make('TrafficManagement-v0')
    
    def smart_policy(observation, info):
        """
        A simple heuristic policy that switches lights based on queue lengths.
        """
        actions = []
        intersection_states = info.get('intersection_states', [])
        
        for i, intersection_state in enumerate(intersection_states):
            if i >= len(intersection_states):
                break
                
            queue_lengths = intersection_state.get('queue_lengths', {})
            current_phase = intersection_state.get('light_phase', 'NS_GREEN')
            
            # Get queue lengths for each direction
            ns_queue = queue_lengths.get('NORTH', 0) + queue_lengths.get('SOUTH', 0)
            ew_queue = queue_lengths.get('EAST', 0) + queue_lengths.get('WEST', 0)
            
            # Simple policy: switch to direction with longer queue
            if current_phase in ['NS_GREEN', 'NS_YELLOW']:
                if ew_queue > ns_queue + 2:  # Add hysteresis
                    actions.append(2)  # Switch to EW_GREEN
                else:
                    actions.append(0)  # Maintain
            else:  # EW_GREEN or EW_YELLOW
                if ns_queue > ew_queue + 2:  # Add hysteresis
                    actions.append(1)  # Switch to NS_GREEN
                else:
                    actions.append(0)  # Maintain
        
        # Pad with maintain actions if needed
        num_intersections = len(intersection_states) if intersection_states else 8
        while len(actions) < num_intersections:
            actions.append(0)
        
        return np.array(actions[:num_intersections])
    
    # Run episodes with smart policy
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        step_count = 0
        
        while True:
            # Smart policy action
            action = smart_policy(observation, info)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print periodic updates
            if step_count % 100 == 0:
                metrics = info.get('metrics', {})
                print(f"  Step {step_count}: "
                      f"Reward={total_reward:.2f}, "
                      f"Vehicles={info['num_vehicles']}, "
                      f"Avg Wait={metrics.get('average_waiting_time', 0):.2f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        final_metrics = info.get('metrics', {})
        print(f"  Final: Steps={step_count}, "
              f"Total Reward={total_reward:.2f}, "
              f"Vehicles Passed={final_metrics.get('total_vehicles_passed', 0)}, "
              f"Avg Queue Length={final_metrics.get('average_queue_length', 0):.2f}")
    
    env.close()
    
    print(f"\nSummary:")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


def environment_analysis():
    """Analyze the environment's behavior and characteristics."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT ANALYSIS")
    print("=" * 60)
    
    env = gym.make('TrafficManagement-v0')
    
    # Analyze observation space
    print(f"Environment Configuration:")
    # Access the underlying environment through the wrapper
    base_env = env.unwrapped
    print(f"  Grid size: {base_env.grid_size}")
    print(f"  Number of intersections: {base_env.num_intersections}")
    print(f"  Maximum vehicles: {base_env.max_vehicles}")
    print(f"  Spawn rate: {base_env.spawn_rate}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Sample observation
    observation, info = env.reset(seed=42)
    print(f"\nObservation Analysis:")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    print(f"  Non-zero elements: {np.count_nonzero(observation)}")
    
    # Analyze reward distribution
    print(f"\nReward Analysis (100 random steps):")
    rewards = []
    
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    rewards = np.array(rewards)
    print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"  Average reward: {rewards.mean():.3f}")
    print(f"  Reward std: {rewards.std():.3f}")
    print(f"  Positive rewards: {(rewards > 0).sum()}/100")
    print(f"  Negative rewards: {(rewards < 0).sum()}/100")
    
    env.close()


def training_example():
    """Show how the environment can be used for RL training."""
    print("\n" + "=" * 60)
    print("TRAINING EXAMPLE (Simplified)")
    print("=" * 60)
    
    env = gym.make('TrafficManagement-v0')
    
    # Simple Q-learning-like approach (for demonstration)
    print("This example shows how you could structure RL training:")
    print("(Note: This is a simplified example, not actual RL training)")
    
    # Initialize simple policy parameters (just for demonstration)
    num_intersections = env.unwrapped.num_intersections
    learning_rate = 0.01
    
    def simple_policy(observation):
        """Simple policy for demonstration - just random with some bias."""
        actions = []
        for i in range(num_intersections):
            # Simple heuristic: slightly favor maintaining current state
            action_probs = [0.5, 0.25, 0.25]  # [maintain, NS_GREEN, EW_GREEN]
            action = np.random.choice(3, p=action_probs)
            actions.append(action)
        
        return np.array(actions)
    
    print("\nRunning simplified training loop...")
    
    episode_rewards = []
    for episode in range(5):
        observation, info = env.reset(seed=episode)
        total_reward = 0
        
        for step in range(200):  # Short episodes for demo
            action = simple_policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Simple policy update (not real RL - just for demonstration)
            # In real RL, you would update neural network weights or Q-values here
            pass  # Skip the update for this demo
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nTraining progress: {episode_rewards}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print("\nFor real RL training, use libraries like Stable-Baselines3:")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO('MlpPolicy', env)")
    print("  model.learn(total_timesteps=100000)")
    
    env.close()


if __name__ == "__main__":
    print("TRAFFIC MANAGEMENT ENVIRONMENT DEMONSTRATION")
    print("This script demonstrates various aspects of the environment.")
    
    # Run demonstrations
    random_rewards, random_lengths = random_policy_demo()
    smart_rewards, smart_lengths = smart_policy_demo()
    
    # Compare policies
    print("\n" + "=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)
    print(f"Random Policy - Avg Reward: {np.mean(random_rewards):.2f}")
    print(f"Smart Policy  - Avg Reward: {np.mean(smart_rewards):.2f}")
    print(f"Improvement: {((np.mean(smart_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")
    
    # Environment analysis
    environment_analysis()
    
    # Training example
    training_example()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("The Traffic Management Environment is ready for RL research!")
    print("=" * 60)