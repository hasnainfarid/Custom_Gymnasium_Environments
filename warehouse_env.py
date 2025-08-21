"""
Warehouse Logistics Gymnasium Environment
A 2D grid-based warehouse environment with robot navigation, package handling, and battery management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import sys

class CellType(Enum):
    EMPTY = 0
    SHELF = 1
    OBSTACLE = 2
    CHARGING_STATION = 3
    PACKAGE_ZONE = 4

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICKUP = 4
    DROPOFF = 5
    CHARGE = 6

class Package:
    def __init__(self, package_id: int, pickup_pos: Tuple[int, int], delivery_pos: Tuple[int, int]):
        self.id = package_id
        self.pickup_pos = pickup_pos
        self.delivery_pos = delivery_pos
        self.picked_up = False
        self.delivered = False

class WarehouseEnv(gym.Env):
    """
    Warehouse Logistics Environment
    
    State Space:
    - Robot position (x, y)
    - Battery level (0-100)
    - Carrying package ID (-1 if none)
    - Package states (pickup/delivery positions, status)
    
    Action Space:
    - 0-3: Move (up, down, left, right)
    - 4: Pickup package
    - 5: Dropoff package
    - 6: Charge battery
    
    Rewards:
    - +100: Successful delivery
    - -1: Each timestep
    - -10: Invalid action
    - -50: Battery death
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, width=15, height=15, num_packages=3, max_battery=100, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.num_packages = num_packages
        self.max_battery = max_battery
        self.render_mode = render_mode
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(7)  # 4 moves + pickup + dropoff + charge
        
        # Observation: robot_x, robot_y, battery, carrying_package_id, packages_info
        # packages_info: for each package: pickup_x, pickup_y, delivery_x, delivery_y, picked_up, delivered
        obs_size = 4 + (num_packages * 6)  # robot(2) + battery(1) + carrying(1) + packages(6*n)
        self.observation_space = spaces.Box(
            low=0, high=max(width, height, max_battery, num_packages),
            shape=(obs_size,), dtype=np.int32
        )
        
        # Initialize grid and components
        self._create_warehouse_layout()
        self.packages = []
        self.robot_pos = None
        self.battery = max_battery
        self.carrying_package = -1
        self.timestep = 0
        self.max_timesteps = 500
        
        # Pygame rendering
        self.window = None
        self.clock = None
        self.cell_size = 40
        self.window_size = (width * self.cell_size, height * self.cell_size + 100)  # Extra space for info
        
        # Colors
        self.colors = {
            'empty': (255, 255, 255),
            'shelf': (139, 69, 19),
            'obstacle': (64, 64, 64),
            'charging_station': (255, 255, 0),
            'package_zone': (144, 238, 144),
            'robot': (0, 0, 255),
            'package': [(255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 165, 0), (0, 255, 255)],
            'delivery': (255, 192, 203)
        }
    
    def _create_warehouse_layout(self):
        """Create the warehouse grid layout"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Add shelves (create aisles)
        for i in range(2, self.height - 2, 3):
            for j in range(2, self.width - 2, 3):
                if random.random() < 0.7:  # 70% chance of shelf
                    self.grid[i:i+2, j:j+2] = CellType.SHELF.value
        
        # Add obstacles randomly
        obstacle_count = int(0.05 * self.width * self.height)
        for _ in range(obstacle_count):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.OBSTACLE.value
        
        # Add charging stations
        charging_positions = [
            (0, 0), (self.width-1, 0), (0, self.height-1), (self.width-1, self.height-1)
        ]
        for x, y in charging_positions:
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.CHARGING_STATION.value
        
        # Add package zones
        package_zone_positions = [
            (self.width//2, 0), (0, self.height//2), (self.width-1, self.height//2), (self.width//2, self.height-1)
        ]
        for x, y in package_zone_positions:
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.PACKAGE_ZONE.value
    
    def _get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty positions in the grid"""
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == CellType.EMPTY.value:
                    empty_positions.append((x, y))
        return empty_positions
    
    def _get_package_zone_positions(self) -> List[Tuple[int, int]]:
        """Get all package zone positions"""
        package_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == CellType.PACKAGE_ZONE.value:
                    package_positions.append((x, y))
        return package_positions
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset robot
        empty_positions = self._get_empty_positions()
        self.robot_pos = random.choice(empty_positions)
        self.battery = self.max_battery
        self.carrying_package = -1
        self.timestep = 0
        
        # Reset packages
        self.packages = []
        package_zone_positions = self._get_package_zone_positions()
        
        for i in range(self.num_packages):
            # Pick random pickup and delivery positions
            pickup_pos = random.choice(package_zone_positions)
            delivery_pos = random.choice(package_zone_positions)
            
            # Ensure pickup and delivery are different
            while delivery_pos == pickup_pos:
                delivery_pos = random.choice(package_zone_positions)
            
            package = Package(i, pickup_pos, delivery_pos)
            self.packages.append(package)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Robot position and state
        obs.extend([self.robot_pos[0], self.robot_pos[1], self.battery, self.carrying_package])
        
        # Package information
        for package in self.packages:
            obs.extend([
                package.pickup_pos[0], package.pickup_pos[1],
                package.delivery_pos[0], package.delivery_pos[1],
                int(package.picked_up), int(package.delivered)
            ])
        
        return np.array(obs, dtype=np.int32)
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid and not blocked"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        cell_type = self.grid[y, x]
        return cell_type != CellType.SHELF.value and cell_type != CellType.OBSTACLE.value
    
    def step(self, action: int):
        """Execute one step in the environment"""
        self.timestep += 1
        reward = -1  # Base timestep penalty
        done = False
        info = {}
        
        # Battery decreases with each action
        self.battery = max(0, self.battery - 1)
        
        # Check for battery death
        if self.battery <= 0:
            reward = -50
            done = True
            info['termination_reason'] = 'battery_death'
            return self._get_observation(), reward, done, False, info
        
        # Execute action
        if action == Action.MOVE_UP.value:
            new_x, new_y = self.robot_pos[0], self.robot_pos[1] - 1
            if self._is_valid_position(new_x, new_y):
                self.robot_pos = (new_x, new_y)
            else:
                reward = -10  # Invalid move penalty
                
        elif action == Action.MOVE_DOWN.value:
            new_x, new_y = self.robot_pos[0], self.robot_pos[1] + 1
            if self._is_valid_position(new_x, new_y):
                self.robot_pos = (new_x, new_y)
            else:
                reward = -10
                
        elif action == Action.MOVE_LEFT.value:
            new_x, new_y = self.robot_pos[0] - 1, self.robot_pos[1]
            if self._is_valid_position(new_x, new_y):
                self.robot_pos = (new_x, new_y)
            else:
                reward = -10
                
        elif action == Action.MOVE_RIGHT.value:
            new_x, new_y = self.robot_pos[0] + 1, self.robot_pos[1]
            if self._is_valid_position(new_x, new_y):
                self.robot_pos = (new_x, new_y)
            else:
                reward = -10
                
        elif action == Action.PICKUP.value:
            if self.carrying_package == -1:  # Not carrying anything
                # Check if there's a package at current position
                for package in self.packages:
                    if (package.pickup_pos == self.robot_pos and 
                        not package.picked_up and not package.delivered):
                        package.picked_up = True
                        self.carrying_package = package.id
                        info['picked_up_package'] = package.id
                        break
                else:
                    reward = -10  # No package to pick up
            else:
                reward = -10  # Already carrying something
                
        elif action == Action.DROPOFF.value:
            if self.carrying_package >= 0:  # Carrying something
                package = self.packages[self.carrying_package]
                if self.robot_pos == package.delivery_pos:
                    # Successful delivery
                    package.delivered = True
                    self.carrying_package = -1
                    reward = 100  # Big reward for delivery
                    info['delivered_package'] = package.id
                else:
                    reward = -10  # Wrong delivery location
            else:
                reward = -10  # Not carrying anything
                
        elif action == Action.CHARGE.value:
            if self.grid[self.robot_pos[1], self.robot_pos[0]] == CellType.CHARGING_STATION.value:
                self.battery = min(self.max_battery, self.battery + 20)  # Charge 20 units
                info['charged'] = True
            else:
                reward = -10  # Not at charging station
        
        # Check win condition
        if all(package.delivered for package in self.packages):
            done = True
            reward += 200  # Bonus for completing all deliveries
            info['termination_reason'] = 'all_delivered'
        
        # Check timeout
        if self.timestep >= self.max_timesteps:
            done = True
            info['termination_reason'] = 'timeout'
        
        return self._get_observation(), reward, done, False, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Warehouse Logistics")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((240, 240, 240))
        
        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                cell_type = CellType(self.grid[y, x])
                if cell_type == CellType.EMPTY:
                    color = self.colors['empty']
                elif cell_type == CellType.SHELF:
                    color = self.colors['shelf']
                elif cell_type == CellType.OBSTACLE:
                    color = self.colors['obstacle']
                elif cell_type == CellType.CHARGING_STATION:
                    color = self.colors['charging_station']
                elif cell_type == CellType.PACKAGE_ZONE:
                    color = self.colors['package_zone']
                
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # Border
        
        # Draw packages
        for package in self.packages:
            if not package.picked_up and not package.delivered:
                # Draw pickup package
                x, y = package.pickup_pos
                center = (x * self.cell_size + self.cell_size // 2,
                         y * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.window, self.colors['package'][package.id % len(self.colors['package'])], 
                                 center, self.cell_size // 4)
            
            # Draw delivery zone
            if not package.delivered:
                x, y = package.delivery_pos
                rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5,
                                 self.cell_size - 10, self.cell_size - 10)
                pygame.draw.rect(self.window, self.colors['delivery'], rect, 3)
        
        # Draw robot
        x, y = self.robot_pos
        center = (x * self.cell_size + self.cell_size // 2,
                 y * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.window, self.colors['robot'], center, self.cell_size // 3)
        
        # If carrying package, draw it on robot
        if self.carrying_package >= 0:
            pygame.draw.circle(self.window, self.colors['package'][self.carrying_package % len(self.colors['package'])], 
                             center, self.cell_size // 6)
        
        # Draw info panel
        info_y = self.height * self.cell_size + 10
        font = pygame.font.Font(None, 24)
        
        # Battery info
        battery_text = font.render(f"Battery: {self.battery}/{self.max_battery}", True, (0, 0, 0))
        self.window.blit(battery_text, (10, info_y))
        
        # Timestep info
        timestep_text = font.render(f"Timestep: {self.timestep}/{self.max_timesteps}", True, (0, 0, 0))
        self.window.blit(timestep_text, (200, info_y))
        
        # Carrying info
        carrying_text = font.render(f"Carrying: {'Package ' + str(self.carrying_package) if self.carrying_package >= 0 else 'None'}", 
                                  True, (0, 0, 0))
        self.window.blit(carrying_text, (400, info_y))
        
        # Packages status
        delivered_count = sum(1 for p in self.packages if p.delivered)
        packages_text = font.render(f"Delivered: {delivered_count}/{len(self.packages)}", True, (0, 0, 0))
        self.window.blit(packages_text, (10, info_y + 30))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    # Quick test
    env = WarehouseEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Warehouse Environment Created!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print("Close the pygame window to exit.")
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            print(f"Episode finished! Reason: {info.get('termination_reason', 'unknown')}")
            break
    
    env.close()