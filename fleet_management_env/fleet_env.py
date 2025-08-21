"""
Fleet Management Environment

A multi-agent reinforcement learning environment for urban delivery logistics
with dynamic traffic, fuel management, and realistic delivery constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


class VehicleType(Enum):
    VAN = "van"
    MOTORCYCLE = "motorcycle" 
    TRUCK = "truck"


class UrgencyLevel(Enum):
    STANDARD = 0
    NORMAL = 1
    URGENT = 2
    CRITICAL = 3


class CustomerZone(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    HOSPITAL = "hospital"


@dataclass
class Vehicle:
    """Vehicle specification and state"""
    vehicle_type: VehicleType
    position: Tuple[int, int]
    fuel: float
    max_fuel: float
    fuel_consumption: float
    capacity: int
    cargo_used: int
    speed: int
    assigned_delivery: int = -1  # -1 if no delivery assigned
    
    def can_move(self) -> bool:
        return self.fuel >= self.fuel_consumption
    
    def move(self, fuel_cost: float = None) -> None:
        if fuel_cost is None:
            fuel_cost = self.fuel_consumption
        self.fuel = max(0, self.fuel - fuel_cost)
    
    def refuel(self) -> None:
        self.fuel = self.max_fuel
    
    def can_pickup(self) -> bool:
        return self.cargo_used < self.capacity and self.assigned_delivery == -1
    
    def pickup_delivery(self, delivery_id: int) -> None:
        if self.can_pickup():
            self.cargo_used += 1
            self.assigned_delivery = delivery_id
    
    def drop_off(self) -> None:
        if self.assigned_delivery != -1:
            self.cargo_used = max(0, self.cargo_used - 1)
            delivered_id = self.assigned_delivery
            self.assigned_delivery = -1
            return delivered_id
        return -1


@dataclass
class DeliveryRequest:
    """Delivery request specification"""
    pickup_location: Tuple[int, int]
    delivery_location: Tuple[int, int]
    urgency: UrgencyLevel
    required_vehicle: Optional[VehicleType]
    time_window_start: int
    time_window_end: int
    deadline: int
    completed: bool = False
    assigned_vehicle: int = -1
    pickup_time: int = -1
    delivery_time: int = -1
    
    def is_available(self, current_time: int) -> bool:
        return (not self.completed and 
                self.time_window_start <= current_time <= self.time_window_end)
    
    def is_overdue(self, current_time: int) -> bool:
        return current_time > self.deadline and not self.completed


class FleetManagementEnv(gym.Env):
    """
    Fleet Management Environment for multi-agent reinforcement learning
    
    Simulates urban delivery logistics with:
    - Multi-vehicle coordination (van, motorcycle, truck)
    - Dynamic traffic patterns
    - Fuel management and refueling stations
    - Time-sensitive deliveries with urgency levels
    - Customer zones with different characteristics
    - Weather effects and seasonal variations
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 25
        self.max_timesteps = 800
        self.depot_position = (12, 12)
        self.fuel_stations = [(5, 5), (20, 5), (5, 20)]
        
        # Vehicle specifications
        self.vehicle_specs = {
            VehicleType.VAN: {"speed": 1, "capacity": 3, "fuel_consumption": 1.0, "range": 80},
            VehicleType.MOTORCYCLE: {"speed": 1, "capacity": 1, "fuel_consumption": 0.5, "range": 120},
            VehicleType.TRUCK: {"speed": 1, "capacity": 5, "fuel_consumption": 2.0, "range": 60}
        }
        
        # Customer zones (quadrants)
        self.customer_zones = {
            CustomerZone.RESIDENTIAL: [(i, j) for i in range(0, 12) for j in range(0, 12)],
            CustomerZone.COMMERCIAL: [(i, j) for i in range(13, 25) for j in range(0, 12)],
            CustomerZone.INDUSTRIAL: [(i, j) for i in range(0, 12) for j in range(13, 25)],
            CustomerZone.HOSPITAL: [(i, j) for i in range(13, 25) for j in range(13, 25)]
        }
        
        # State space: 45 elements total
        # - Vehicle positions (3 * 2 = 6)
        # - Vehicle fuel levels (3 * 1 = 3)  
        # - Vehicle cargo used (3 * 1 = 3)
        # - Current deliveries assigned (3 * 1 = 3)
        # - Delivery locations (12 * 2 = 24) 
        # - Delivery urgency (12 * 1 = 12)
        # - Traffic congestion (5 * 5 = 25) - reduced grid for traffic
        
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size, 4), 
            shape=(87,), dtype=np.float32
        )
        
        # Action space: MultiDiscrete for 3 vehicles, 8 actions each
        # 0: Stay, 1-4: Move (Up/Down/Left/Right), 5: Pick up, 6: Drop off, 7: Refuel
        self.action_space = spaces.MultiDiscrete([8, 8, 8])
        
        # Initialize environment state
        self.vehicles = []
        self.delivery_requests = []
        self.traffic_grid = np.zeros((5, 5))  # Simplified traffic grid
        self.timestep = 0
        self.weather_effect = 1.0
        self.total_reward = 0
        self.completed_deliveries = 0
        self.missed_deadlines = 0
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (1000, 800)
        
        # Performance tracking
        self.episode_stats = {
            "deliveries_completed": 0,
            "fuel_consumed": 0,
            "total_distance": 0,
            "customer_satisfaction": 0,
            "urgent_deliveries_missed": 0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset timestep and stats
        self.timestep = 0
        self.total_reward = 0
        self.completed_deliveries = 0
        self.missed_deadlines = 0
        self.weather_effect = 1.0
        
        # Initialize vehicles at depot
        vehicle_types = [VehicleType.VAN, VehicleType.MOTORCYCLE, VehicleType.TRUCK]
        self.vehicles = []
        
        for i, vtype in enumerate(vehicle_types):
            specs = self.vehicle_specs[vtype]
            vehicle = Vehicle(
                vehicle_type=vtype,
                position=self.depot_position,
                fuel=specs["range"],
                max_fuel=specs["range"],
                fuel_consumption=specs["fuel_consumption"],
                capacity=specs["capacity"],
                cargo_used=0,
                speed=specs["speed"]
            )
            self.vehicles.append(vehicle)
        
        # Generate delivery requests
        self._generate_delivery_requests()
        
        # Initialize traffic
        self._update_traffic()
        
        # Reset episode stats
        self.episode_stats = {
            "deliveries_completed": 0,
            "fuel_consumed": 0,
            "total_distance": 0,
            "customer_satisfaction": 0,
            "urgent_deliveries_missed": 0
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute actions for all vehicles and return new state"""
        if len(action) != 3:
            raise ValueError("Action must be provided for all 3 vehicles")
        
        rewards = []
        info = {}
        
        # Execute actions for each vehicle
        for i, (vehicle, vehicle_action) in enumerate(zip(self.vehicles, action)):
            reward = self._execute_vehicle_action(vehicle, vehicle_action, i)
            rewards.append(reward)
        
        # Update environment state
        self.timestep += 1
        
        # Update traffic every 50 timesteps
        if self.timestep % 50 == 0:
            self._update_traffic()
        
        # Update weather every 100 timesteps
        if self.timestep % 100 == 0:
            self._update_weather()
        
        # Check for missed deadlines
        self._check_missed_deadlines()
        
        # Calculate total reward
        total_reward = sum(rewards)
        self.total_reward += total_reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.timestep >= self.max_timesteps
        
        observation = self._get_observation()
        info = self._get_info()
        info["individual_rewards"] = rewards
        info["episode_stats"] = self.episode_stats.copy()
        
        return observation, total_reward, terminated, truncated, info
    
    def _execute_vehicle_action(self, vehicle: Vehicle, action: int, vehicle_id: int) -> float:
        """Execute action for a single vehicle and return reward"""
        reward = 0
        
        # Base timestep cost for active vehicles
        if vehicle.fuel > 0:
            reward -= 2
        
        if action == 0:  # Stay
            pass
        
        elif 1 <= action <= 4:  # Move
            if vehicle.can_move():
                old_pos = vehicle.position
                new_pos = self._get_new_position(vehicle.position, action)
                
                if self._is_valid_position(new_pos):
                    # Calculate fuel cost based on traffic
                    traffic_cost = self._get_traffic_cost(new_pos)
                    fuel_cost = vehicle.fuel_consumption * self.weather_effect * traffic_cost
                    
                    vehicle.position = new_pos
                    vehicle.move(fuel_cost)
                    
                    # Traffic penalty
                    if traffic_cost > 1.5:
                        reward -= 5
                    
                    # Update distance traveled
                    self.episode_stats["total_distance"] += 1
                    self.episode_stats["fuel_consumed"] += fuel_cost
                else:
                    reward -= 10  # Invalid move penalty
            else:
                reward -= 50  # Out of fuel penalty
        
        elif action == 5:  # Pick up delivery
            pickup_reward = self._attempt_pickup(vehicle, vehicle_id)
            reward += pickup_reward
        
        elif action == 6:  # Drop off delivery  
            dropoff_reward = self._attempt_dropoff(vehicle, vehicle_id)
            reward += dropoff_reward
        
        elif action == 7:  # Refuel
            refuel_reward = self._attempt_refuel(vehicle)
            reward += refuel_reward
        
        else:
            reward -= 10  # Invalid action
        
        return reward
    
    def _get_new_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate new position based on movement action"""
        x, y = position
        if action == 1:  # Up
            return (x, max(0, y - 1))
        elif action == 2:  # Down
            return (x, min(self.grid_size - 1, y + 1))
        elif action == 3:  # Left
            return (max(0, x - 1), y)
        elif action == 4:  # Right
            return (min(self.grid_size - 1, x + 1), y)
        return position
    
    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _get_traffic_cost(self, position: Tuple[int, int]) -> float:
        """Get traffic multiplier for fuel consumption at position"""
        # Map grid position to traffic grid (25x25 -> 5x5)
        tx = min(4, position[0] // 5)
        ty = min(4, position[1] // 5)
        traffic_level = self.traffic_grid[ty, tx]
        
        if traffic_level == 0:  # Light traffic
            return 1.0
        elif traffic_level == 1:  # Medium traffic
            return 1.5
        else:  # Heavy traffic
            return 2.0
    
    def _attempt_pickup(self, vehicle: Vehicle, vehicle_id: int) -> float:
        """Attempt to pick up a delivery at current position"""
        if not vehicle.can_pickup():
            return -10  # Can't pickup
        
        # Find available deliveries at current position
        available_deliveries = []
        for i, delivery in enumerate(self.delivery_requests):
            if (delivery.pickup_location == vehicle.position and 
                delivery.is_available(self.timestep) and
                delivery.assigned_vehicle == -1):
                
                # Check vehicle compatibility
                if (delivery.required_vehicle is None or 
                    delivery.required_vehicle == vehicle.vehicle_type):
                    available_deliveries.append((i, delivery))
        
        if not available_deliveries:
            return -10  # No valid deliveries
        
        # Pick up the most urgent delivery
        delivery_idx, delivery = max(available_deliveries, 
                                   key=lambda x: x[1].urgency.value)
        
        vehicle.pickup_delivery(delivery_idx)
        delivery.assigned_vehicle = vehicle_id
        delivery.pickup_time = self.timestep
        
        return 20  # Pickup reward
    
    def _attempt_dropoff(self, vehicle: Vehicle, vehicle_id: int) -> float:
        """Attempt to drop off current delivery"""
        if vehicle.assigned_delivery == -1:
            return -10  # No delivery to drop off
        
        delivery = self.delivery_requests[vehicle.assigned_delivery]
        
        if vehicle.position != delivery.delivery_location:
            return -10  # Wrong location
        
        # Calculate delivery reward
        delivered_id = vehicle.drop_off()
        delivery.completed = True
        delivery.delivery_time = self.timestep
        
        # Base reward based on urgency
        urgency_rewards = {
            UrgencyLevel.STANDARD: 50,
            UrgencyLevel.NORMAL: 100,
            UrgencyLevel.URGENT: 200,
            UrgencyLevel.CRITICAL: 200
        }
        
        reward = urgency_rewards[delivery.urgency]
        
        # Time bonus/penalty
        delivery_time = self.timestep - delivery.pickup_time
        optimal_distance = abs(delivery.pickup_location[0] - delivery.delivery_location[0]) + \
                          abs(delivery.pickup_location[1] - delivery.delivery_location[1])
        
        if delivery_time <= optimal_distance + 2:  # Efficient delivery
            reward += 15
        
        if self.timestep <= delivery.deadline:
            # On time delivery
            self.episode_stats["deliveries_completed"] += 1
            self.episode_stats["customer_satisfaction"] += 1
        else:
            # Late delivery penalty
            penalty_multiplier = delivery.urgency.value + 1
            reward -= 20 * penalty_multiplier
            self.episode_stats["urgent_deliveries_missed"] += 1
        
        self.completed_deliveries += 1
        return reward
    
    def _attempt_refuel(self, vehicle: Vehicle) -> float:
        """Attempt to refuel at current position"""
        if vehicle.position in self.fuel_stations:
            if vehicle.fuel < vehicle.max_fuel:
                vehicle.refuel()
                return 10  # Small refuel bonus
            else:
                return -5  # Already full
        else:
            return -10  # Not at fuel station
    
    def _generate_delivery_requests(self):
        """Generate 8-12 delivery requests with various properties"""
        num_requests = np.random.randint(8, 13)
        self.delivery_requests = []
        
        zone_positions = {}
        for zone, positions in self.customer_zones.items():
            zone_positions[zone] = positions
        
        for i in range(num_requests):
            # Random pickup and delivery locations
            pickup_zone = np.random.choice(list(CustomerZone))
            delivery_zone = np.random.choice(list(CustomerZone))
            
            pickup_pos = random.choice(zone_positions[pickup_zone])
            delivery_pos = random.choice(zone_positions[delivery_zone])
            
            # Ensure pickup and delivery are different
            while pickup_pos == delivery_pos:
                delivery_pos = random.choice(zone_positions[delivery_zone])
            
            # Assign urgency
            urgency = np.random.choice(list(UrgencyLevel), 
                                     p=[0.3, 0.4, 0.2, 0.1])  # Most are standard/normal
            
            # Vehicle requirements based on delivery zone
            required_vehicle = None
            if delivery_zone == CustomerZone.HOSPITAL:
                required_vehicle = VehicleType.MOTORCYCLE  # Fast medical deliveries
            elif delivery_zone == CustomerZone.INDUSTRIAL:
                if np.random.random() < 0.6:  # 60% need truck for heavy cargo
                    required_vehicle = VehicleType.TRUCK
            
            # Time windows
            if delivery_zone == CustomerZone.COMMERCIAL:
                # Business hours
                time_start = np.random.randint(50, 200)  # 9 AM equivalent
                time_end = min(time_start + 300, 600)    # 5 PM equivalent
            else:
                # 24/7 availability
                time_start = 0
                time_end = self.max_timesteps
            
            # Deadline based on urgency
            base_time = abs(pickup_pos[0] - delivery_pos[0]) + abs(pickup_pos[1] - delivery_pos[1])
            deadline_multipliers = {
                UrgencyLevel.STANDARD: 4.0,
                UrgencyLevel.NORMAL: 3.0,
                UrgencyLevel.URGENT: 2.0,
                UrgencyLevel.CRITICAL: 1.5
            }
            
            deadline = int(base_time * deadline_multipliers[urgency]) + 50
            
            delivery = DeliveryRequest(
                pickup_location=pickup_pos,
                delivery_location=delivery_pos,
                urgency=urgency,
                required_vehicle=required_vehicle,
                time_window_start=time_start,
                time_window_end=time_end,
                deadline=deadline
            )
            
            self.delivery_requests.append(delivery)
    
    def _update_traffic(self):
        """Update traffic congestion patterns"""
        # Create traffic hotspots that change over time
        self.traffic_grid = np.random.choice([0, 1, 2], size=(5, 5), p=[0.6, 0.3, 0.1])
        
        # Add some persistent congestion in commercial areas (top-right)
        self.traffic_grid[0:2, 3:5] = np.random.choice([1, 2], size=(2, 2), p=[0.4, 0.6])
    
    def _update_weather(self):
        """Update weather effects on fuel consumption"""
        weather_conditions = [0.8, 1.0, 1.2, 1.5]  # Good, normal, rain, storm
        probabilities = [0.3, 0.5, 0.15, 0.05]
        self.weather_effect = np.random.choice(weather_conditions, p=probabilities)
    
    def _check_missed_deadlines(self):
        """Check for missed delivery deadlines"""
        for delivery in self.delivery_requests:
            if delivery.is_overdue(self.timestep) and not delivery.completed:
                if delivery.urgency in [UrgencyLevel.URGENT, UrgencyLevel.CRITICAL]:
                    self.missed_deadlines += 1
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # All deliveries completed
        if all(d.completed for d in self.delivery_requests):
            return True
        
        # All vehicles out of fuel
        if all(v.fuel <= 0 for v in self.vehicles):
            return True
        
        # Too many urgent deliveries missed (50% threshold)
        urgent_deliveries = sum(1 for d in self.delivery_requests 
                              if d.urgency in [UrgencyLevel.URGENT, UrgencyLevel.CRITICAL])
        if urgent_deliveries > 0 and self.missed_deadlines >= urgent_deliveries * 0.5:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        obs = []
        
        # Vehicle positions (6 values)
        for vehicle in self.vehicles:
            obs.extend([vehicle.position[0], vehicle.position[1]])
        
        # Vehicle fuel levels (3 values)
        for vehicle in self.vehicles:
            obs.append(vehicle.fuel / vehicle.max_fuel)  # Normalized
        
        # Vehicle cargo used (3 values) 
        for vehicle in self.vehicles:
            obs.append(vehicle.cargo_used / vehicle.capacity)  # Normalized
        
        # Current deliveries assigned (3 values)
        for vehicle in self.vehicles:
            obs.append(vehicle.assigned_delivery if vehicle.assigned_delivery != -1 else -1)
        
        # Delivery request locations (24 values) - pad to 12 requests
        for i in range(12):
            if i < len(self.delivery_requests) and not self.delivery_requests[i].completed:
                delivery = self.delivery_requests[i]
                obs.extend([delivery.pickup_location[0], delivery.pickup_location[1]])
            else:
                obs.extend([-1, -1])  # No delivery or completed
        
        # Delivery urgency levels (12 values)
        for i in range(12):
            if i < len(self.delivery_requests) and not self.delivery_requests[i].completed:
                obs.append(self.delivery_requests[i].urgency.value)
            else:
                obs.append(-1)  # No delivery or completed
        
        # Traffic congestion map (25 values)
        obs.extend(self.traffic_grid.flatten())
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information"""
        active_deliveries = sum(1 for d in self.delivery_requests if not d.completed)
        vehicles_with_fuel = sum(1 for v in self.vehicles if v.fuel > 0)
        
        return {
            "timestep": self.timestep,
            "active_deliveries": active_deliveries,
            "completed_deliveries": self.completed_deliveries,
            "vehicles_with_fuel": vehicles_with_fuel,
            "weather_effect": self.weather_effect,
            "total_reward": self.total_reward,
            "missed_deadlines": self.missed_deadlines
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Fleet Management Environment")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface(self.window_size)
        canvas.fill((240, 240, 240))  # Light gray background
        
        # Calculate cell size for grid
        grid_width = 600
        cell_size = grid_width // self.grid_size
        grid_offset_x = 50
        grid_offset_y = 50
        
        # Draw customer zones
        zone_colors = {
            CustomerZone.RESIDENTIAL: (200, 255, 200),  # Light green
            CustomerZone.COMMERCIAL: (200, 200, 255),   # Light blue  
            CustomerZone.INDUSTRIAL: (255, 255, 200),   # Light yellow
            CustomerZone.HOSPITAL: (255, 200, 200)      # Light red
        }
        
        for zone, positions in self.customer_zones.items():
            color = zone_colors[zone]
            for pos in positions:
                rect = pygame.Rect(
                    grid_offset_x + pos[0] * cell_size,
                    grid_offset_y + pos[1] * cell_size,
                    cell_size, cell_size
                )
                pygame.draw.rect(canvas, color, rect)
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(canvas, (150, 150, 150),
                           (grid_offset_x + i * cell_size, grid_offset_y),
                           (grid_offset_x + i * cell_size, grid_offset_y + grid_width))
            # Horizontal lines  
            pygame.draw.line(canvas, (150, 150, 150),
                           (grid_offset_x, grid_offset_y + i * cell_size),
                           (grid_offset_x + grid_width, grid_offset_y + i * cell_size))
        
        # Draw traffic congestion
        for ty in range(5):
            for tx in range(5):
                traffic_level = self.traffic_grid[ty, tx]
                if traffic_level > 0:
                    # Map traffic grid back to main grid
                    start_x = tx * 5
                    start_y = ty * 5
                    
                    color = (255, 255, 0, 100) if traffic_level == 1 else (255, 0, 0, 150)
                    
                    for dx in range(5):
                        for dy in range(5):
                            if start_x + dx < self.grid_size and start_y + dy < self.grid_size:
                                rect = pygame.Rect(
                                    grid_offset_x + (start_x + dx) * cell_size,
                                    grid_offset_y + (start_y + dy) * cell_size,
                                    cell_size, cell_size
                                )
                                # Create surface with alpha for transparency
                                s = pygame.Surface((cell_size, cell_size))
                                s.set_alpha(100 if traffic_level == 1 else 150)
                                s.fill(color[:3])
                                canvas.blit(s, rect)
        
        # Draw fuel stations
        for station in self.fuel_stations:
            center = (
                grid_offset_x + station[0] * cell_size + cell_size // 2,
                grid_offset_y + station[1] * cell_size + cell_size // 2
            )
            pygame.draw.circle(canvas, (0, 100, 0), center, cell_size // 3)
            pygame.draw.circle(canvas, (255, 255, 255), center, cell_size // 4)
        
        # Draw depot
        depot_rect = pygame.Rect(
            grid_offset_x + self.depot_position[0] * cell_size,
            grid_offset_y + self.depot_position[1] * cell_size,
            cell_size, cell_size
        )
        pygame.draw.rect(canvas, (100, 100, 100), depot_rect)
        
        # Draw delivery requests
        urgency_colors = {
            UrgencyLevel.STANDARD: (255, 255, 255),
            UrgencyLevel.NORMAL: (255, 255, 0),
            UrgencyLevel.URGENT: (255, 165, 0),
            UrgencyLevel.CRITICAL: (255, 0, 0)
        }
        
        for delivery in self.delivery_requests:
            if not delivery.completed:
                # Pickup location (circle)
                pickup_center = (
                    grid_offset_x + delivery.pickup_location[0] * cell_size + cell_size // 2,
                    grid_offset_y + delivery.pickup_location[1] * cell_size + cell_size // 2
                )
                color = urgency_colors[delivery.urgency]
                pygame.draw.circle(canvas, color, pickup_center, cell_size // 4)
                pygame.draw.circle(canvas, (0, 0, 0), pickup_center, cell_size // 4, 2)
                
                # Delivery location (diamond)
                delivery_center = (
                    grid_offset_x + delivery.delivery_location[0] * cell_size + cell_size // 2,
                    grid_offset_y + delivery.delivery_location[1] * cell_size + cell_size // 2
                )
                diamond_points = [
                    (delivery_center[0], delivery_center[1] - cell_size // 4),
                    (delivery_center[0] + cell_size // 4, delivery_center[1]),
                    (delivery_center[0], delivery_center[1] + cell_size // 4),
                    (delivery_center[0] - cell_size // 4, delivery_center[1])
                ]
                pygame.draw.polygon(canvas, color, diamond_points)
                pygame.draw.polygon(canvas, (0, 0, 0), diamond_points, 2)
        
        # Draw vehicles
        vehicle_colors = {
            VehicleType.VAN: (0, 0, 255),        # Blue
            VehicleType.MOTORCYCLE: (0, 255, 0), # Green  
            VehicleType.TRUCK: (255, 0, 0)       # Red
        }
        
        for i, vehicle in enumerate(self.vehicles):
            center = (
                grid_offset_x + vehicle.position[0] * cell_size + cell_size // 2,
                grid_offset_y + vehicle.position[1] * cell_size + cell_size // 2
            )
            color = vehicle_colors[vehicle.vehicle_type]
            
            if vehicle.vehicle_type == VehicleType.VAN:
                # Square for van
                rect = pygame.Rect(center[0] - cell_size//3, center[1] - cell_size//3, 
                                 2*cell_size//3, 2*cell_size//3)
                pygame.draw.rect(canvas, color, rect)
            elif vehicle.vehicle_type == VehicleType.MOTORCYCLE:
                # Circle for motorcycle
                pygame.draw.circle(canvas, color, center, cell_size // 3)
            else:  # TRUCK
                # Rectangle for truck
                rect = pygame.Rect(center[0] - cell_size//2, center[1] - cell_size//4,
                                 cell_size, cell_size//2)
                pygame.draw.rect(canvas, color, rect)
            
            # Fuel indicator
            fuel_ratio = vehicle.fuel / vehicle.max_fuel
            fuel_color = (255, 0, 0) if fuel_ratio < 0.3 else (255, 255, 0) if fuel_ratio < 0.6 else (0, 255, 0)
            fuel_rect = pygame.Rect(center[0] - cell_size//3, center[1] + cell_size//2,
                                  int(2*cell_size//3 * fuel_ratio), 4)
            pygame.draw.rect(canvas, fuel_color, fuel_rect)
        
        # Draw metrics panel
        panel_x = grid_width + 100
        panel_y = 50
        font = pygame.font.Font(None, 24)
        
        metrics = [
            f"Timestep: {self.timestep}/{self.max_timesteps}",
            f"Total Reward: {self.total_reward:.1f}",
            f"Deliveries: {self.completed_deliveries}/{len(self.delivery_requests)}",
            f"Weather Effect: {self.weather_effect:.1f}x",
            "",
            "Vehicle Status:",
        ]
        
        for i, vehicle in enumerate(self.vehicles):
            metrics.append(f"{vehicle.vehicle_type.value.title()}: Fuel {vehicle.fuel:.1f}/{vehicle.max_fuel}")
            metrics.append(f"  Pos: {vehicle.position}, Cargo: {vehicle.cargo_used}/{vehicle.capacity}")
        
        for i, metric in enumerate(metrics):
            text = font.render(metric, True, (0, 0, 0))
            canvas.blit(text, (panel_x, panel_y + i * 25))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up rendering resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()