#!/usr/bin/env python3
"""
Fleet Management Environment - Demo Script

This script demonstrates basic usage of the Fleet Management Environment.
Run this after installing the package and its dependencies.

Usage:
    python demo.py [--render] [--steps N]
"""

import argparse
import time
import numpy as np

try:
    import gymnasium as gym
    from fleet_env import FleetManagementEnv
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Dependencies not available: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    DEPS_AVAILABLE = False


class SimpleGreedyAgent:
    """A simple greedy agent for demonstration"""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.depot_position = (12, 12)
        self.fuel_stations = [(5, 5), (20, 5), (5, 20)]
    
    def get_action(self, observation):
        """Get action based on simple greedy strategy"""
        actions = []
        
        # Parse observation (simplified)
        vehicle_positions = []
        for i in range(3):
            x = int(observation[i * 2])
            y = int(observation[i * 2 + 1])
            vehicle_positions.append((x, y))
        
        fuel_levels = observation[6:9]
        assigned_deliveries = observation[12:15]
        
        for i, (pos, fuel, assigned) in enumerate(zip(vehicle_positions, fuel_levels, assigned_deliveries)):
            # Simple decision logic
            if fuel < 0.2:  # Low fuel
                # Find nearest fuel station
                nearest_station = min(self.fuel_stations, 
                                    key=lambda fs: abs(fs[0] - pos[0]) + abs(fs[1] - pos[1]))
                
                if pos == nearest_station:
                    actions.append(7)  # Refuel
                else:
                    # Move towards fuel station
                    actions.append(self._move_towards(pos, nearest_station))
            
            elif int(assigned) >= 0:  # Has delivery
                # Try to drop off (simplified - would need actual delivery location)
                actions.append(6)  # Attempt drop off
            
            else:  # No delivery, look for pickups
                actions.append(5)  # Attempt pickup
        
        return actions
    
    def _move_towards(self, current, target):
        """Simple movement towards target"""
        if target[0] > current[0]:
            return 4  # Right
        elif target[0] < current[0]:
            return 3  # Left
        elif target[1] > current[1]:
            return 2  # Down
        elif target[1] < current[1]:
            return 1  # Up
        else:
            return 0  # Stay


def run_demo(render=True, max_steps=200):
    """Run a demonstration episode"""
    
    if not DEPS_AVAILABLE:
        print("Cannot run demo - dependencies not installed")
        return
    
    print("Fleet Management Environment - Demo")
    print("=" * 50)
    
    # Create environment
    render_mode = "human" if render else None
    env = FleetManagementEnv(render_mode=render_mode)
    
    # Create agent
    agent = SimpleGreedyAgent(env.action_space)
    
    print(f"Environment created:")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Vehicles: {len(env.vehicles)}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Reset environment
    observation, info = env.reset(seed=42)
    
    print(f"\nEpisode started:")
    print(f"  Delivery requests: {info['active_deliveries']}")
    print(f"  Vehicles with fuel: {info['vehicles_with_fuel']}")
    
    # Run episode
    total_reward = 0
    step = 0
    
    try:
        while step < max_steps:
            # Get action from agent
            action = agent.get_action(observation)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print progress every 20 steps
            if step % 20 == 0 or terminated or truncated:
                print(f"Step {step:3d}: reward={reward:6.1f}, "
                      f"total={total_reward:7.1f}, "
                      f"deliveries={info['completed_deliveries']}/{info['active_deliveries'] + info['completed_deliveries']}, "
                      f"fuel_vehicles={info['vehicles_with_fuel']}")
            
            # Render if enabled
            if render and step % 5 == 0:  # Render every 5 steps to slow down
                time.sleep(0.2)
            
            # Check termination
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"\nEpisode ended ({reason}) at step {step}")
                break
        
        # Final statistics
        print(f"\n" + "=" * 50)
        print(f"Episode Summary:")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Deliveries completed: {info['completed_deliveries']}")
        print(f"  Vehicles with fuel: {info['vehicles_with_fuel']}")
        print(f"  Average reward per step: {total_reward/step:.2f}")
        
        if info['completed_deliveries'] > 0:
            print(f"  Success! Completed {info['completed_deliveries']} deliveries")
        else:
            print(f"  No deliveries completed - try different strategy")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        env.close()
        print("Demo completed")


def main():
    """Main demo function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Fleet Management Environment Demo')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Enable visualization (default: True)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--steps', type=int, default=200,
                       help='Maximum number of steps (default: 200)')
    
    args = parser.parse_args()
    
    # Handle render flags
    render = args.render and not args.no_render
    
    if not DEPS_AVAILABLE:
        print("Demo cannot run without dependencies.")
        print("Install with: pip install -r requirements.txt")
        return 1
    
    try:
        run_demo(render=render, max_steps=args.steps)
        return 0
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())