import gymnasium as gym
import numpy as np
import time
from snake_env import SnakeEnvClassic

def test_random_agent():
    """Test the environment with a random agent."""
    env = SnakeEnvClassic(render_mode="human")
    observation, info = env.reset()
    
    total_reward = 0
    episode_count = 0
    
    for episode in range(5):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        
        print(f"Episode {episode + 1}")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            env.render()
            time.sleep(0.1)  # Slow down for visualization
            
            if done:
                print(f"Episode finished with score: {info['score']}")
                break
        
        total_reward += episode_reward
        episode_count += 1
    
    print(f"Average reward: {total_reward / episode_count}")
    env.close()

def test_manual_control():
    """Test the environment with manual keyboard control."""
    env = SnakeEnvClassic(render_mode="human")
    observation, info = env.reset()
    
    print("Manual control mode:")
    print("Use arrow keys to control the snake")
    print("Press 'q' to quit")
    
    done = False
    while not done:
        # Handle pygame events
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                else:
                    continue
                
                observation, reward, done, truncated, info = env.step(action)
                print(f"Score: {info['score']}, Reward: {reward}")
        
        env.render()
        time.sleep(0.1)
    
    env.close()

if __name__ == "__main__":
    print("Snake Environment Test")
    print("1. Random agent test")
    print("2. Manual control test")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        test_random_agent()
    elif choice == "2":
        test_manual_control()
    else:
        print("Invalid choice. Running random agent test.")
        test_random_agent() 