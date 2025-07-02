import numpy as np
from environment import BusSystemEnv

if __name__ == "__main__":
    # Choose visualization mode
    print("Bus System Environment Test")
    print("1. CLI only (fast)")
    print("2. With Pygame visualization (demo mode)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    enable_viz = choice == "2"
    demo_speed = 0.5 if enable_viz else 1.0  # Slower for demo
    
    env = BusSystemEnv(render_mode=None, enable_visualization=enable_viz, demo_speed=demo_speed)
    obs, info = env.reset()
    
    if not enable_viz:
        print("Initial State:")
        print(env.get_state_summary())

    total_reward = 0
    steps = 60 if enable_viz else 60  # More steps for demo
    
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if not enable_viz:
            print(f"\nStep {t+1}")
            print(f"Action: {action}")
            print(f"Reward: {reward:.2f}")
            
            # Debug information
            print("\nDEBUG INFO:")
            for i, bus in enumerate(env.buses):
                print(f"  Bus {i}: Stop {bus.current_stop}, Stopped: {bus.is_stopped}, "
                      f"Remaining: {bus.remaining_time}, Dwell: {bus.dwell_time}, "
                      f"Passengers: {len(bus.onboard_passengers)}")
            
            print(env.get_state_summary())
        
        # Render visualization if enabled
        if enable_viz:
            env.render()
        
        if terminated or truncated:
            print("Episode finished.")
            break

    print(f"\nTotal reward after {t+1} steps: {total_reward:.2f}")
    env.close() 