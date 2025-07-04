#!/usr/bin/env python3
"""
Test script for Explosive Turret Environment
"""

import time
import math
from explosive_turret_env import TurretEnv


def random_agent(env, num_episodes=5):
    """Run a random agent for the specified number of episodes."""
    print(f"Starting Explosive Turret Environment - {num_episodes} episodes")
    print("=" * 50)
    
    total_reward = 0
    successful_hits = 0
    perfect_destructions = 0
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        env.current_episode = episode + 1
        observation, info = env.reset()
        
        episode_reward = 0
        step = 0
        
        while True:
            # Render the environment
            env.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            # Take random action
            action = env.action_space.sample()
            angle, force = action
            
            if step == 0:  # Only print once per episode
                print(f"  Firing: Angle={angle*180/math.pi:.1f}°, Force={force:.1f}N")
            
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Check results
            if done:
                if info['target_hit']:
                    print(f"  TARGET HIT! Explosion triggered!")
                    successful_hits += 1
                    
                    # Enhanced reward analysis
                    if hasattr(env, 'impact_velocity'):
                        print(f"  Impact Velocity: {env.impact_velocity:.1f} m/s")
                    if hasattr(env, 'pieces_affected_count'):
                        print(f"  Pieces Affected: {env.pieces_affected_count}/20")
                    if hasattr(env, 'total_piece_velocity'):
                        print(f"  Total Piece Velocity: {env.total_piece_velocity:.1f} m/s")
                    
                    if info['total_reward'] >= 80:
                        print(f"  PERFECT DESTRUCTION! Complete annihilation!")
                        perfect_destructions += 1
                    elif info['total_reward'] >= 50:
                        print(f"  GREAT HIT! Massive destruction!")
                    elif info['total_reward'] >= 20:
                        print(f"  GOOD HIT! Significant damage!")
                    else:
                        print(f"  WEAK HIT! Minimal damage.")
                elif info['shell_out_of_bounds']:
                    print(f"  MISS! Shell went out of bounds")
                else:
                    print(f"  MISS! Shell landed without hitting target")
                
                print(f"  Episode reward: {episode_reward:.1f}")
                break
            
            # Small delay to see the trajectory
            time.sleep(0.016)
        
        total_reward += episode_reward
        
        # Show result for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(0.016)
    
    print("\n" + "=" * 50)
    print(f"Training Complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Successful hits: {successful_hits} ({successful_hits/num_episodes*100:.1f}%)")
    print(f"Perfect destructions: {perfect_destructions}")
    print(f"Average reward: {total_reward/num_episodes:.2f}")
    
    # Keep window open
    print("\nPress CTRL+C or close window to exit.")
    try:
        while True:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(0.016)
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == "__main__":
    import pygame
    
    # Create and run the environment
    env = TurretEnv(render_mode="human")
    
    try:
        random_agent(env, num_episodes=5)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        env.close()
    except Exception as e:
        print(f"\nError during training: {e}")
        env.close() 