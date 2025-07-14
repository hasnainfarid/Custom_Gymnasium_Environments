#!/usr/bin/env python3
"""
Restaurant Environment Example

This script demonstrates basic usage of the restaurant environment,
including running episodes, collecting statistics, and visualization.
"""

import numpy as np
import random
from restaurant_env import RestaurantEnv
from visualization import RestaurantVisualization
from utils import calculate_episode_statistics, create_performance_report

def run_random_episode(env, max_steps=500):
    """Run a single episode with random actions"""
    obs, info = env.reset()
    episode_data = []
    total_reward = 0
    
    print(f"Starting episode...")
    print(f"Initial state: {info['waiting_customers']} waiting customers, {info['idle_waiters']} idle waiters")
    
    for step in range(max_steps):
        # Choose random action
        action = env.action_space.sample()
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Store step data
        episode_data.append({
            'timestep': step,
            'action': action,
            'reward': reward,
            'info': info
        })
        
        # Print progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}, "
                  f"Waiting={info['waiting_customers']}, Idle={info['idle_waiters']}")
        
        if terminated or truncated:
            break
    
    print(f"Episode finished after {len(episode_data)} steps")
    print(f"Final total reward: {total_reward:.2f}")
    print(f"Final stats: {info['episode_stats']}")
    
    return episode_data, total_reward, info

def run_simple_policy_episode(env, max_steps=500):
    """Run episode with a simple heuristic policy"""
    obs, info = env.reset()
    episode_data = []
    total_reward = 0
    
    print(f"Starting episode with simple policy...")
    
    for step in range(max_steps):
        # Simple policy: prioritize seating customers, then serving food, then cleaning
        action = 3  # Default to do nothing
        
        # Check if we can seat a customer
        if info['waiting_customers'] > 0 and info['idle_waiters'] > 0:
            # Find available tables
            available_tables = sum(1 for table in env.tables if table.is_available_for_seating())
            if available_tables > 0:
                action = 0  # Seat customer
        
        # Check if we can serve food
        elif info['ready_orders'] > 0 and info['idle_waiters'] > 0:
            action = 1  # Serve food
        
        # Check if we can clean tables
        elif info['dirty_tables'] > 0 and info['idle_waiters'] > 0:
            action = 2  # Clean table
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Store step data
        episode_data.append({
            'timestep': step,
            'action': action,
            'reward': reward,
            'info': info
        })
        
        if terminated or truncated:
            break
    
    print(f"Simple policy episode finished after {len(episode_data)} steps")
    print(f"Final total reward: {total_reward:.2f}")
    
    return episode_data, total_reward, info

def demonstrate_visualization():
    """Demonstrate the visualization system"""
    print("\n" + "="*60)
    print("DEMONSTRATING VISUALIZATION")
    print("="*60)
    
    # Create environment with visualization
    env = RestaurantEnv(render_mode='human')
    vis = RestaurantVisualization(env)
    
    # Run a short episode with visualization
    obs, info = env.reset()
    vis.render(obs, info)
    
    print("Running short episode with visualization...")
    print("Close the visualization window to continue...")
    
    for step in range(50):  # Short episode for demo
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render current state
        vis.render(obs, info)
        
        if terminated or truncated:
            break
    
    vis.close()
    env.close()

def analyze_episode_performance(episode_data):
    """Analyze and report episode performance"""
    print("\n" + "="*60)
    print("EPISODE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    stats = calculate_episode_statistics(episode_data)
    
    # Print key metrics
    print(f"Total Reward: {stats['total_reward']:.2f}")
    print(f"Average Reward per Step: {stats['average_reward_per_step']:.3f}")
    print(f"Efficiency Score: {stats['efficiency_score']:.3f}")
    print(f"Customers Served: {stats['customers_served']}")
    print(f"Customers Left: {stats['customers_left']}")
    print(f"Customer Satisfaction Rate: {stats['customer_satisfaction_rate']:.1%}")
    print(f"Average Wait Time: {stats['average_wait_time']:.1f} timesteps")
    
    # Generate detailed report
    report = create_performance_report(episode_data)
    print("\n" + report)

def main():
    """Main demonstration function"""
    print("Restaurant Environment Demonstration")
    print("="*60)
    
    # Create environment
    env = RestaurantEnv()
    
    # Run random episode
    print("\n1. Running episode with random actions...")
    episode_data, total_reward, final_info = run_random_episode(env)
    
    # Analyze performance
    analyze_episode_performance(episode_data)
    
    # Run simple policy episode
    print("\n2. Running episode with simple policy...")
    episode_data2, total_reward2, final_info2 = run_simple_policy_episode(env)
    
    # Compare policies
    print("\n" + "="*60)
    print("POLICY COMPARISON")
    print("="*60)
    print(f"Random Policy: {total_reward:.2f} total reward")
    print(f"Simple Policy: {total_reward2:.2f} total reward")
    
    if total_reward2 > total_reward:
        print("Simple policy performed better!")
    else:
        print("Random policy performed better (or similar)")
    
    # Demonstrate visualization (optional)
    try:
        demonstrate_visualization()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to display issues or pygame not being available")
    
    # Clean up
    env.close()
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main() 