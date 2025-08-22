#!/usr/bin/env python3
"""
Simple test script to verify the snake environment visualization works.
"""

import time
from snake_env import SnakeEnvClassic

def test_visualization():
    """Test the pygame visualization."""
    print("Testing Snake Environment Visualization")
    print("You should see a window with the snake game.")
    print("The snake will move randomly for 10 seconds.")
    print("Close the window to exit.")
    
    # Create environment with human rendering
    env = SnakeEnvClassic(render_mode="human", grid_size=15)
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial score: {info['score']}")
    
    # Run for a few steps
    for step in range(50):
        # Take random action
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Small delay
        time.sleep(0.2)
        
        if done:
            print(f"Game ended with score: {info['score']}")
            break
    
    print("Test completed!")
    env.close()

if __name__ == "__main__":
    test_visualization() 