"""
Test script for Warehouse Logistics Environment
Demonstrates functionality with visualization, manual control, and automated testing.
"""

import pygame
import numpy as np
import time
from warehouse_env import WarehouseEnv, Action
import random

class WarehouseTester:
    def __init__(self):
        self.env = None
        self.running = True
        
    def run_manual_control(self):
        """Run environment with manual keyboard control"""
        print("\n=== Manual Control Mode ===")
        print("Controls:")
        print("  Arrow Keys: Move robot")
        print("  P: Pickup package")
        print("  D: Dropoff package")
        print("  C: Charge battery")
        print("  R: Reset environment")
        print("  ESC: Exit")
        print("  SPACE: Pause/Unpause")
        
        env = WarehouseEnv(width=12, height=12, num_packages=3, render_mode="human")
        obs, info = env.reset(seed=42)
        
        clock = pygame.time.Clock()
        paused = False
        total_reward = 0
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    action = None
                    
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("Environment reset!")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Unpaused")
                    elif not paused:
                        if event.key == pygame.K_UP:
                            action = Action.MOVE_UP.value
                        elif event.key == pygame.K_DOWN:
                            action = Action.MOVE_DOWN.value
                        elif event.key == pygame.K_LEFT:
                            action = Action.MOVE_LEFT.value
                        elif event.key == pygame.K_RIGHT:
                            action = Action.MOVE_RIGHT.value
                        elif event.key == pygame.K_p:
                            action = Action.PICKUP.value
                        elif event.key == pygame.K_d:
                            action = Action.DROPOFF.value
                        elif event.key == pygame.K_c:
                            action = Action.CHARGE.value
                    
                    if action is not None:
                        obs, reward, done, truncated, info = env.step(action)
                        total_reward += reward
                        
                        if reward != -1:  # Don't print timestep penalties
                            print(f"Action: {Action(action).name}, Reward: {reward}, Total: {total_reward}")
                        
                        if 'picked_up_package' in info:
                            print(f"Picked up package {info['picked_up_package']}!")
                        if 'delivered_package' in info:
                            print(f"Delivered package {info['delivered_package']}!")
                        if 'charged' in info:
                            print("Battery charged!")
                        
                        if done:
                            print(f"\nEpisode finished!")
                            print(f"Reason: {info.get('termination_reason', 'unknown')}")
                            print(f"Total reward: {total_reward}")
                            print("Press 'R' to reset or ESC to exit")
            
            env.render()
            clock.tick(10)
        
        env.close()
    
    def run_automated_test(self):
        """Run automated test with simple AI agent"""
        print("\n=== Automated Test Mode ===")
        
        env = WarehouseEnv(width=10, height=10, num_packages=2, render_mode="human")
        
        # Test multiple episodes
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            obs, info = env.reset(seed=episode)
            total_reward = 0
            steps = 0
            
            while steps < 200:  # Max steps per episode
                # Simple AI: random valid actions with some logic
                action = self._choose_action(env, obs)
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Print important events
                if 'picked_up_package' in info:
                    print(f"Step {steps}: Picked up package {info['picked_up_package']}")
                if 'delivered_package' in info:
                    print(f"Step {steps}: Delivered package {info['delivered_package']}")
                if 'charged' in info:
                    print(f"Step {steps}: Charged battery")
                
                env.render()
                time.sleep(0.1)  # Slow down for visualization
                
                if done:
                    print(f"Episode {episode + 1} finished in {steps} steps")
                    print(f"Reason: {info.get('termination_reason', 'unknown')}")
                    print(f"Total reward: {total_reward}")
                    time.sleep(2)  # Pause between episodes
                    break
            
            if not done:
                print(f"Episode {episode + 1} timed out after {steps} steps")
                print(f"Total reward: {total_reward}")
        
        env.close()
    
    def _choose_action(self, env, obs):
        """Simple AI logic for choosing actions"""
        robot_x, robot_y = int(obs[0]), int(obs[1])
        battery = int(obs[2])
        carrying = int(obs[3])
        
        # If battery is low, try to charge
        if battery < 20:
            # Check if at charging station
            if env.grid[robot_y, robot_x] == 3:  # Charging station
                return Action.CHARGE.value
            else:
                # Move towards nearest charging station
                charging_stations = [(0, 0), (env.width-1, 0), (0, env.height-1), (env.width-1, env.height-1)]
                closest_station = min(charging_stations, 
                                    key=lambda pos: abs(pos[0] - robot_x) + abs(pos[1] - robot_y))
                return self._move_towards(robot_x, robot_y, closest_station[0], closest_station[1])
        
        # If carrying package, try to deliver
        if carrying >= 0:
            package = env.packages[carrying]
            delivery_x, delivery_y = package.delivery_pos
            
            if (robot_x, robot_y) == (delivery_x, delivery_y):
                return Action.DROPOFF.value
            else:
                return self._move_towards(robot_x, robot_y, delivery_x, delivery_y)
        
        # If not carrying, try to pick up package
        for package in env.packages:
            if not package.picked_up and not package.delivered:
                pickup_x, pickup_y = package.pickup_pos
                
                if (robot_x, robot_y) == (pickup_x, pickup_y):
                    return Action.PICKUP.value
                else:
                    return self._move_towards(robot_x, robot_y, pickup_x, pickup_y)
        
        # Default: random movement
        return random.choice([Action.MOVE_UP.value, Action.MOVE_DOWN.value, 
                            Action.MOVE_LEFT.value, Action.MOVE_RIGHT.value])
    
    def _move_towards(self, from_x, from_y, to_x, to_y):
        """Choose movement action to get closer to target"""
        dx = to_x - from_x
        dy = to_y - from_y
        
        if abs(dx) > abs(dy):
            return Action.MOVE_RIGHT.value if dx > 0 else Action.MOVE_LEFT.value
        else:
            return Action.MOVE_DOWN.value if dy > 0 else Action.MOVE_UP.value
    
    def run_performance_test(self):
        """Run performance test without visualization"""
        print("\n=== Performance Test Mode ===")
        
        env = WarehouseEnv(width=15, height=15, num_packages=5)
        
        episodes = 100
        total_rewards = []
        success_count = 0
        
        print(f"Running {episodes} episodes...")
        
        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 300:
                action = env.action_space.sample()  # Random actions
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    if info.get('termination_reason') == 'all_delivered':
                        success_count += 1
                    break
            
            total_rewards.append(total_reward)
            
            if (episode + 1) % 20 == 0:
                print(f"Completed {episode + 1}/{episodes} episodes")
        
        # Statistics
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = success_count / episodes * 100
        
        print(f"\nPerformance Results:")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Success rate: {success_rate:.1f}% ({success_count}/{episodes})")
        print(f"Best reward: {max(total_rewards)}")
        print(f"Worst reward: {min(total_rewards)}")
        
        env.close()
    
    def run_feature_demo(self):
        """Demonstrate specific features of the environment"""
        print("\n=== Feature Demonstration ===")
        
        env = WarehouseEnv(width=8, height=8, num_packages=2, render_mode="human")
        obs, info = env.reset(seed=123)
        
        print("Demonstrating key features:")
        print("1. Environment layout with different cell types")
        print("2. Robot movement and collision detection")
        print("3. Package pickup and delivery")
        print("4. Battery management and charging")
        
        # Demo sequence
        demo_actions = [
            (Action.MOVE_RIGHT, "Moving right"),
            (Action.MOVE_DOWN, "Moving down"),
            (Action.PICKUP, "Attempting pickup (should fail - no package)"),
            (Action.CHARGE, "Attempting charge (should fail - not at station)"),
            (Action.MOVE_UP, "Moving up"),
            (Action.MOVE_LEFT, "Moving left"),
        ]
        
        for action, description in demo_actions:
            print(f"\n{description}")
            obs, reward, done, truncated, info = env.step(action.value)
            print(f"Reward: {reward}")
            if info:
                print(f"Info: {info}")
            
            env.render()
            time.sleep(1.5)
            
            if done:
                break
        
        print("\nDemo complete! Environment will continue running...")
        time.sleep(3)
        env.close()


def main():
    """Main function to run different test modes"""
    print("Warehouse Logistics Environment Test Suite")
    print("==========================================")
    
    tester = WarehouseTester()
    
    while True:
        print("\nSelect test mode:")
        print("1. Manual Control (keyboard)")
        print("2. Automated Test (simple AI)")
        print("3. Performance Test (no visualization)")
        print("4. Feature Demo")
        print("5. Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                tester.run_manual_control()
            elif choice == '2':
                tester.run_automated_test()
            elif choice == '3':
                tester.run_performance_test()
            elif choice == '4':
                tester.run_feature_demo()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()