"""
Test script for the Smart Parking Lot RL Environment with minute-based system.
"""

# Paperspace-compatible imports
import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from core.parking_env import SmartParkingEnv
    from utils.visualization import ParkingVisualization
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    raise


import numpy as np
import random
from typing import List, Dict
import time


def run_simple_test():
    """Run simple test without visualization."""
    print("=== Simple Test Mode ===")
    
    # Create environment
    env = SmartParkingEnv()
    observation, info = env.reset()
    
    print(f"Environment created!")
    print(f"Observation shape: {observation.shape}")
    print(f"Starting time: {info['hour']:02d}:{info['minute']:02d}")
    print(f"Total timesteps in episode: 1440 (24 hours * 60 minutes)")
    print()
    
    # Run first 100 timesteps for testing
    total_reward = 0.0
    step = 0
    max_steps = 100
    
    print(f"Running first {max_steps} timesteps...")
    print("Step | Action | Reward | Revenue | Time  | Queue")
    print("-" * 55)
    
    while step < max_steps:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Print step info every 10 steps
        if step % 10 == 0 or step < 10:
            action_name = ["Idle", "Assign_A", "Assign_B", "Assign_C", "Reject", 
                          "Price_A", "Price_B", "Price_C"][action]
            print(f"{step:4d} | {action_name:8s} | {reward:6.2f} | ${info['total_revenue']:7.2f} | {info['hour']:02d}:{info['minute']:02d} | {info['queue_length']:5d}")
        
        step += 1
        
        if terminated or truncated:
            break
    
    print("-" * 55)
    print(f"Test completed after {step} timesteps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final revenue: ${info['total_revenue']:.2f}")
    print(f"Total rejections: {info['rejections']}")
    print(f"Average reward per timestep: {total_reward/step:.3f}")
    
    env.close()


def run_pygame_visualization():
    """Run pygame visualization for first few hours."""
    print("=== Pygame Visualization Test Mode ===")
    
    try:
        # Create environment
        env = SmartParkingEnv()
        observation, info = env.reset()
        
        # Get actual components from environment
        parking_lot = env.parking_lot
        pricing_manager = env.pricing_manager
        
        print("Initializing pygame visualization...")
        print("Controls:")
        print("- SPACE: Pause/Resume")
        print("- 1,2,3,4: Speed control (1x,2x,3x,5x)")
        print("- ESC: Quit")
        print()
        
        # Create visualization with actual environment components
        with ParkingVisualization(parking_lot, pricing_manager) as viz:
            print("Pygame window opened! Running simulation...")
            
            # Run simulation for first 2 hours (120 timesteps)
            max_steps = 120
            step = 0
            
            while step < max_steps:
                # Sample action
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Get actual customers from environment queue
                customers = list(env.parking_lot.queue)
                customer_stats = env.customer_manager.get_statistics()
                
                # Show visualization (faster for testing)
                viz_start_time = time.time()
                while time.time() - viz_start_time < 0.1:  # Show each step for 0.1 seconds
                    if not viz.run_visualization(info['hour'], info['minute'], customers, 
                                               customer_stats, reward):
                        print("Visualization stopped by user.")
                        return
                
                step += 1
                
                # Print progress every 30 steps (30 minutes)
                if step % 30 == 0:
                    print(f"Time: {info['hour']:02d}:{info['minute']:02d}, Step: {step}, Revenue: ${info['total_revenue']:.2f}")
                
                if terminated or truncated:
                    break
                    
        print("Pygame visualization completed!")
        env.close()
        
    except Exception as e:
        print(f"Pygame visualization failed: {e}")
        print("Make sure pygame is installed: pip install pygame")


def run_complete_episode_with_pygame():
    """Run complete 24-hour episode (1440 timesteps) with pygame visualization."""
    print("=== Complete 24-Hour Episode with Pygame Visualization ===")
    print("Warning: This will run 1440 timesteps (24 hours). Use speed controls!")
    
    proceed = input("Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        return
    
    try:
        # Create environment
        env = SmartParkingEnv()
        observation, info = env.reset()
        
        # Get actual components from environment
        parking_lot = env.parking_lot
        pricing_manager = env.pricing_manager
        
        print("Initializing pygame visualization...")
        print("Controls: SPACE (pause), 1-4 (speed), ESC (quit)")
        print("Tip: Use speed 4 (5x) for faster simulation!")
        print()
        
        # Run complete episode with pygame
        with ParkingVisualization(parking_lot, pricing_manager) as viz:
            total_reward = 0.0
            step = 0
            
            print("Running 24-hour episode (1440 timesteps)...")
            print("Time  | Step | Action    | Reward | Revenue | Queue")
            print("-" * 60)
            
            while step < 1440:  # Complete 24-hour episode
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                
                # Print progress every hour (60 steps)
                if step % 60 == 0:
                    action_names = ["Idle", "Assign_A", "Assign_B", "Assign_C", "Reject", 
                                  "Price_A", "Price_B", "Price_C"]
                    action_name = action_names[action]
                    print(f"{info['hour']:02d}:{info['minute']:02d} | {step:4d} | {action_name:9s} | {reward:6.2f} | ${info['total_revenue']:7.2f} | {info['queue_length']:5d}")
                
                # Get actual customers from environment
                customers = list(env.parking_lot.queue)
                customer_stats = env.customer_manager.get_statistics()
                
                # Run pygame visualization (very fast)
                viz_start_time = time.time()
                while time.time() - viz_start_time < 0.05:  # Show each step for 0.05 seconds
                    if not viz.run_visualization(info['hour'], info['minute'], customers, 
                                               customer_stats, reward):
                        print("Visualization stopped by user.")
                        break
                
                step += 1
                
                if terminated or truncated:
                    break
            
            print("-" * 60)
            print(f"Episode completed in {step} timesteps")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final revenue: ${info['total_revenue']:.2f}")
            print(f"Total rejections: {info['rejections']}")
            print(f"Average reward per timestep: {total_reward/step:.3f}")
        
        env.close()
        
    except Exception as e:
        print(f"Complete episode with pygame failed: {e}")
        print("Make sure pygame is installed: pip install pygame")


def run_action_space_test():
    """Test all action types to verify they work correctly."""
    print("=== Action Space Test Mode ===")
    
    env = SmartParkingEnv()
    observation, info = env.reset()
    
    print(f"Environment created!")
    print(f"Action space: {env.action_space}")
    print(f"Total possible actions: {env.action_space.n}")
    print()
    
    # Test each action type
    action_names = ["Idle", "Assign_A", "Assign_B", "Assign_C", "Reject", 
                   "Price_A", "Price_B", "Price_C"]
    
    print("Testing each action type...")
    print("Action | Name      | Reward | Revenue | Queue | Description")
    print("-" * 70)
    
    for action in range(8):
        observation, reward, terminated, truncated, info = env.step(action)
        
        description = ""
        if action == 0:
            description = "No action taken"
        elif action in [1, 2, 3]:
            description = f"Assign customer to zone {['A', 'B', 'C'][action-1]}"
        elif action == 4:
            description = "Reject customer"
        else:
            description = f"Toggle price for zone {['A', 'B', 'C'][action-5]}"
        
        print(f"{action:6d} | {action_names[action]:9s} | {reward:6.2f} | ${info['total_revenue']:7.2f} | {info['queue_length']:5d} | {description}")
    
    print()
    print("Action space test completed!")
    env.close()


def main():
    """Main test function with options."""
    print("Smart Parking Lot RL Environment - Minute-Based Test Script")
    print("=" * 65)
    print("Choose test mode:")
    print("0 - Simple test (first 100 timesteps, no visualization)")
    print("1 - Pygame visualization test (first 2 hours)")
    print("2 - Complete 24-hour episode with pygame visualization")
    print("3 - Action space test (test all 8 action types)")
    print()
    
    try:
        choice = input("Enter your choice (0, 1, 2, or 3): ").strip()
        
        if choice == "0":
            run_simple_test()
        elif choice == "1":
            run_pygame_visualization()
        elif choice == "2":
            run_complete_episode_with_pygame()
        elif choice == "3":
            run_action_space_test()
        else:
            print("Invalid choice. Running simple test...")
            run_simple_test()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main() 