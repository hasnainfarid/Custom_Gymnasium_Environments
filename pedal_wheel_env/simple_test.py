"""
Simple test script to verify the visualization works.
"""

import numpy as np
import time
from environment import PedalWheelEnv

def simple_visualization_test():
    """Simple test to verify visualization works."""
    print("Starting simple visualization test...")
    
    # Create environment with human rendering
    env = PedalWheelEnv(render_mode="human")
    observation, info = env.reset()
    
    print("Environment created and reset successfully!")
    print(f"Initial observation: {observation}")
    print(f"Screen size: {env.renderer.screen.get_size() if env.renderer else 'No renderer'}")
    
    # Run a few steps with simple actions
    for step in range(50):
        # Simple forward motion
        action = np.array([0.2, 0.2], dtype=np.float32)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Pos={info['position']:.1f}m, Vel={info['velocity']:.1f}m/s, Tilt={np.degrees(info['tilt_angle']):.1f}°")
        
        if terminated or truncated:
            print("Episode ended!")
            break
        
        time.sleep(0.1)  # Slow down to see what's happening
    
    env.close()
    print("Test completed!")

if __name__ == "__main__":
    simple_visualization_test() 