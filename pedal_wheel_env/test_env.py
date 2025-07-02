"""
Test script for the Pedal Wheel Environment.
Demonstrates the environment with various control schemes.
"""

import numpy as np
import time
from environment import PedalWheelEnv


def test_random_actions():
    """Test the environment with random actions."""
    print("=== Testing Random Actions ===")
    
    env = PedalWheelEnv(render_mode="human")
    observation, info = env.reset()
    
    total_reward = 0
    step_count = 0
    
    try:
        while step_count < 200:  # Run for 200 steps
            # Generate random actions
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print info every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Position={info['position']:.1f}m, "
                      f"Velocity={info['velocity']:.1f}m/s, "
                      f"Tilt={np.degrees(info['tilt_angle']):.1f}°, "
                      f"Reward={reward:.2f}")
            
            # Check if episode ended
            if terminated:
                print(f"Episode ended: Unicycle fell over at step {step_count}")
                break
            elif truncated:
                print(f"Episode ended: Max steps reached ({step_count})")
                break
            
            # Small delay for visualization
            time.sleep(0.05)
    
    finally:
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final position: {info['position']:.1f}m")
        print(f"Energy used: {info['energy_used']:.1f}J")
        env.close()


def test_fixed_actions():
    """Test the environment with fixed action patterns."""
    print("\n=== Testing Fixed Action Patterns ===")
    
    env = PedalWheelEnv(render_mode="human")
    observation, info = env.reset()
    
    # Define some fixed action patterns
    patterns = [
        # Pattern 1: Gentle forward motion
        lambda step: np.array([0.3, 0.3], dtype=np.float32),
        
        # Pattern 2: Alternating pedals
        lambda step: np.array([0.5 if step % 20 < 10 else -0.2, 
                              0.2 if step % 20 < 10 else 0.5], dtype=np.float32),
        
        # Pattern 3: Balancing pattern
        lambda step: np.array([0.1 * np.sin(step * 0.1), 
                              0.1 * np.cos(step * 0.1)], dtype=np.float32),
    ]
    
    current_pattern = 0
    total_reward = 0
    step_count = 0
    
    try:
        while step_count < 300:  # Run for 300 steps
            # Switch patterns every 100 steps
            if step_count % 100 == 0 and step_count > 0:
                current_pattern = (current_pattern + 1) % len(patterns)
                print(f"Switching to pattern {current_pattern + 1}")
            
            # Get action from current pattern
            action = patterns[current_pattern](step_count)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print info every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Pattern {current_pattern + 1}, "
                      f"Position={info['position']:.1f}m, "
                      f"Velocity={info['velocity']:.1f}m/s, "
                      f"Tilt={np.degrees(info['tilt_angle']):.1f}°")
            
            # Check if episode ended
            if terminated:
                print(f"Episode ended: Unicycle fell over at step {step_count}")
                break
            elif truncated:
                print(f"Episode ended: Max steps reached ({step_count})")
                break
            
            # Small delay for visualization
            time.sleep(0.05)
    
    finally:
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final position: {info['position']:.1f}m")
        env.close()


def test_manual_control():
    """Test the environment with manual keyboard control."""
    print("\n=== Testing Manual Control ===")
    print("Controls:")
    print("  W/S: Left pedal up/down")
    print("  Up/Down arrows: Right pedal up/down")
    print("  Q: Quit")
    print("  R: Reset")
    
    env = PedalWheelEnv(render_mode="human")
    observation, info = env.reset()
    
    # Initialize pygame for keyboard input
    import pygame
    pygame.init()
    
    left_pedal = 0.0
    right_pedal = 0.0
    pedal_step = 0.1
    
    total_reward = 0
    step_count = 0
    
    try:
        while step_count < 1000:
            # Handle keyboard events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_r:
                        observation, info = env.reset()
                        left_pedal = 0.0
                        right_pedal = 0.0
                        total_reward = 0
                        step_count = 0
                        print("Environment reset!")
            
            # Handle continuous key presses
            keys = pygame.key.get_pressed()
            
            # Left pedal control (W/S)
            if keys[pygame.K_w]:
                left_pedal = min(1.0, left_pedal + pedal_step)
            elif keys[pygame.K_s]:
                left_pedal = max(-1.0, left_pedal - pedal_step)
            else:
                # Gradual return to zero
                left_pedal *= 0.9
            
            # Right pedal control (Up/Down arrows)
            if keys[pygame.K_UP]:
                right_pedal = min(1.0, right_pedal + pedal_step)
            elif keys[pygame.K_DOWN]:
                right_pedal = max(-1.0, right_pedal - pedal_step)
            else:
                # Gradual return to zero
                right_pedal *= 0.9
            
            # Take step
            action = np.array([left_pedal, right_pedal], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print info every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}: L={left_pedal:.2f}, R={right_pedal:.2f}, "
                      f"Pos={info['position']:.1f}m, Vel={info['velocity']:.1f}m/s, "
                      f"Tilt={np.degrees(info['tilt_angle']):.1f}°")
            
            # Check if episode ended
            if terminated:
                print(f"Episode ended: Unicycle fell over at step {step_count}")
                observation, info = env.reset()
                left_pedal = 0.0
                right_pedal = 0.0
                total_reward = 0
                step_count = 0
            elif truncated:
                print(f"Episode ended: Max steps reached ({step_count})")
                observation, info = env.reset()
                left_pedal = 0.0
                right_pedal = 0.0
                total_reward = 0
                step_count = 0
            
            # Small delay for visualization
            time.sleep(0.05)
    
    finally:
        print(f"Final total reward: {total_reward:.2f}")
        env.close()
        pygame.quit()


def test_observation_space():
    """Test the observation space and action space."""
    print("\n=== Testing Observation and Action Spaces ===")
    
    env = PedalWheelEnv()
    observation, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {observation}")
    print(f"Observation shape: {observation.shape}")
    print(f"Observation dtype: {observation.dtype}")
    
    # Test action validation
    valid_action = np.array([0.5, -0.3], dtype=np.float32)
    invalid_action = np.array([1.5, 0.0], dtype=np.float32)
    
    print(f"Valid action {valid_action} in action space: {env.action_space.contains(valid_action)}")
    print(f"Invalid action {invalid_action} in action space: {env.action_space.contains(invalid_action)}")
    
    env.close()


def test_physics_consistency():
    """Test physics consistency and bounds."""
    print("\n=== Testing Physics Consistency ===")
    
    env = PedalWheelEnv()
    observation, info = env.reset()
    
    # Test that physics values stay within reasonable bounds
    for step in range(100):
        action = np.array([0.1, 0.1], dtype=np.float32)  # Small forward force
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Check bounds
        assert -1.0 <= observation[0] <= 1.0, f"Position out of bounds: {observation[0]}"
        assert -1.0 <= observation[1] <= 1.0, f"Velocity out of bounds: {observation[1]}"
        assert -1.0 <= observation[2] <= 1.0, f"Tilt out of bounds: {observation[2]}"
        assert -1.0 <= observation[3] <= 1.0, f"Angular velocity out of bounds: {observation[3]}"
        assert -1.0 <= observation[4] <= 1.0, f"Wheel angular velocity out of bounds: {observation[4]}"
        
        if terminated:
            break
    
    print("Physics consistency test passed!")
    env.close()


if __name__ == "__main__":
    print("Pedal Wheel Environment Test Suite")
    print("=" * 40)
    
    try:
        # Run all tests
        test_observation_space()
        test_physics_consistency()
        test_random_actions()
        test_fixed_actions()
        test_manual_control()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll tests completed!") 