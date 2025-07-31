#!/usr/bin/env python3
"""
Simple demo script for the World Builder Game environment.
"""

import time
import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.world_builder_env import WorldBuilderEnv


def manual_control():
    """Run the environment with manual control."""
    env = WorldBuilderEnv(render_mode="human")
    obs, info = env.reset()
    
    print("World Builder Game - Manual Control")
    print("Click buttons to take actions. Game pauses between actions.")
    
    total_reward = 0.0
    running = True
    while running:
        # Render and get action from buttons
        action = env.render()
        
        # Check for quit
        if action == -1:
            running = False
            break
        
        # If action was taken, step the environment
        if action is not None:
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            print(f"Action: {action}, Reward: {reward:.1f}, Total Reward: {total_reward:.1f}")
            print(f"Resources: {info['resources']}")
            print(f"Population: {info['population']}/{info['population_capacity']}")
            print(f"Steps: {info['steps']}, Win Steps: {info['win_steps']}")
            print("-" * 50)
            
            if done:
                if info['population'] <= 0:
                    print("Game Over! Population died.")
                elif info['win_steps'] >= 50:
                    print("Congratulations! You won!")
                else:
                    print("Game Over!")
                running = False
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.05)
    
    env.close()


def random_agent():
    """Run the environment with a random agent."""
    env = WorldBuilderEnv(render_mode="human")
    obs, info = env.reset()
    
    print("World Builder Game - Random Agent")
    print("Watch the AI play automatically. Close window to stop.")
    
    total_reward = 0.0
    running = True
    while running:
        # Render and check for quit
        action = env.render()
        if action == -1:
            running = False
            break
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"Action: {action}, Reward: {reward:.1f}, Total Reward: {total_reward:.1f}")
        print(f"Resources: {info['resources']}")
        print(f"Population: {info['population']}/{info['population_capacity']}")
        print(f"Steps: {info['steps']}, Win Steps: {info['win_steps']}")
        print("-" * 50)
        
        if done:
            if info['population'] <= 0:
                print("Game Over! Population died.")
            elif info['win_steps'] >= 50:
                print("Congratulations! You won!")
            else:
                print("Game Over!")
            running = False
        
        time.sleep(0.5)  # Slower for random agent
    
    env.close()


if __name__ == "__main__":
    print("World Builder Game Demo")
    print("1. Manual Control")
    print("2. Random Agent")
    
    choice = input("Choose mode (1-2): ").strip()
    
    if choice == "1":
        manual_control()
    elif choice == "2":
        random_agent()
    else:
        print("Invalid choice. Running manual control...")
        manual_control() 