"""
Debug test script to check physics and visualization.
"""

import numpy as np
import time
from environment import PedalWheelEnv

def debug_physics_test():
    """Debug test to check physics calculations."""
    print("Starting debug physics test...")
    
    # Create environment without rendering first
    env = PedalWheelEnv(render_mode=None)
    observation, info = env.reset()
    
    print("Initial state:")
    state = env.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Test with stronger actions
    for step in range(20):
        # Use stronger action to see movement
        action = np.array([0.8, 0.8], dtype=np.float32)  # Strong forward force
        
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.get_state()
        
        print(f"\nStep {step}:")
        print(f"  Action: {action}")
        print(f"  Position: {state['x']:.3f}m")
        print(f"  Velocity: {state['vx']:.3f}m/s")
        print(f"  Wheel omega: {state['wheel_omega']:.3f}rad/s")
        print(f"  Tilt: {np.degrees(state['theta']):.1f}°")
        print(f"  Reward: {reward:.3f}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()

def debug_visualization_test():
    """Debug test with visualization."""
    print("\nStarting debug visualization test...")
    
    env = PedalWheelEnv(render_mode="human")
    observation, info = env.reset()
    
    print("Environment created with visualization!")
    
    # Test with stronger actions and slower pace
    for step in range(30):
        # Use stronger action
        action = np.array([0.5, 0.5], dtype=np.float32)
        
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.get_state()
        
        print(f"Step {step}: Pos={state['x']:.2f}m, Vel={state['vx']:.2f}m/s, Tilt={np.degrees(state['theta']):.1f}°")
        
        if terminated or truncated:
            print("Episode ended!")
            break
        
        time.sleep(0.2)  # Slower to see what's happening
    
    env.close()

if __name__ == "__main__":
    debug_physics_test()
    debug_visualization_test() 