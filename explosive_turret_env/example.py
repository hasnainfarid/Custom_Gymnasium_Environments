#!/usr/bin/env python3
"""
Simple example of using the Explosive Turret Environment
"""

import math
import gymnasium as gym
from explosive_turret_env import TurretEnv


def main():
    print("Explosive Turret Environment Example")
    print("=" * 40)
    
    # Method 1: Direct import
    print("\nMethod 1: Direct Import")
    env1 = TurretEnv(render_mode="human")
    run_episode(env1, "Direct Import")
    env1.close()
    
    # Method 2: Using Gymnasium
    print("\nMethod 2: Using Gymnasium")
    env2 = gym.make("ExplosiveTurret-v1", render_mode="human")
    run_episode(env2, "Gymnasium")
    env2.close()


def run_episode(env, method_name):
    """Run a single episode with the given environment."""
    print(f"\nRunning episode with {method_name}...")
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial observation: {observation}")
    
    # Take a few actions
    for i in range(3):
        # Fire at different angles and forces
        angle = math.pi / 4 + (i * 0.2)  # 45°, 65°, 85°
        force = 50.0 + (i * 20)  # 50N, 70N, 90N
        
        action = [angle, force]
        print(f"\nFiring: Angle={angle*180/math.pi:.1f}°, Force={force:.1f}N")
        
        observation, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}")
        print(f"Target hit: {info['target_hit']}")
        
        if done:
            print("Episode finished!")
            break


if __name__ == "__main__":
    main() 