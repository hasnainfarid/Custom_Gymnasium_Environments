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
        self.window_size = (1100, 700)
        
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
    
    def _draw_gradient_background(self, surface):
        """Draw a gradient background"""
        width, height = surface.get_size()
        # Sky blue to light gray gradient
        for y in range(height):
            ratio = y / height
            r = int(135 + (240 - 135) * ratio)  # 135 to 240
            g = int(206 + (240 - 206) * ratio)  # 206 to 240  
            b = int(235 + (240 - 235) * ratio)  # 235 to 240
            color = (r, g, b)
            pygame.draw.line(surface, color, (0, y), (width, y))
    
    def _draw_3d_rect(self, surface, color, rect, depth=3):
        """Draw a 3D-looking rectangle with depth"""
        # Main rectangle
        pygame.draw.rect(surface, color, rect)
        
        # Highlight (top and left)
        highlight_color = tuple(min(255, c + 40) for c in color)
        pygame.draw.line(surface, highlight_color, 
                        (rect.left, rect.top), (rect.right-1, rect.top), 2)
        pygame.draw.line(surface, highlight_color,
                        (rect.left, rect.top), (rect.left, rect.bottom-1), 2)
        
        # Shadow (bottom and right)
        shadow_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.line(surface, shadow_color,
                        (rect.left+1, rect.bottom-1), (rect.right-1, rect.bottom-1), 2)
        pygame.draw.line(surface, shadow_color,
                        (rect.right-1, rect.top+1), (rect.right-1, rect.bottom-1), 2)
    
    def _draw_3d_circle(self, surface, color, center, radius):
        """Draw a 3D-looking circle with gradient"""
        # Create gradient effect
        for i in range(radius, 0, -1):
            ratio = i / radius
            # Lighter towards center
            r = min(255, int(color[0] + (255 - color[0]) * (1 - ratio) * 0.3))
            g = min(255, int(color[1] + (255 - color[1]) * (1 - ratio) * 0.3))
            b = min(255, int(color[2] + (255 - color[2]) * (1 - ratio) * 0.3))
            pygame.draw.circle(surface, (r, g, b), center, i)
        
        # Highlight
        highlight_center = (center[0] - radius//3, center[1] - radius//3)
        pygame.draw.circle(surface, (255, 255, 255), highlight_center, radius//4)
    
    def _draw_building(self, surface, rect, color, building_type="generic"):
        """Draw a 3D building with depth"""
        # Base building
        self._draw_3d_rect(surface, color, rect, depth=4)
        
        # Add building details based on type
        if building_type == "residential":
            # Add windows
            window_size = max(2, rect.width // 8)
            for i in range(2):
                for j in range(2):
                    window_x = rect.x + rect.width//4 + i * rect.width//2
                    window_y = rect.y + rect.height//4 + j * rect.height//3
                    window_rect = pygame.Rect(window_x, window_y, window_size, window_size)
                    pygame.draw.rect(surface, (100, 150, 200), window_rect)
        
        elif building_type == "commercial":
            # Add glass windows
            window_width = rect.width - 4
            window_height = max(2, rect.height // 4)
            for i in range(3):
                window_y = rect.y + 2 + i * (rect.height // 3)
                window_rect = pygame.Rect(rect.x + 2, window_y, window_width, window_height)
                pygame.draw.rect(surface, (150, 200, 255), window_rect)
        
        elif building_type == "industrial":
            # Add industrial details
            pygame.draw.rect(surface, (80, 80, 80), 
                           (rect.x + rect.width//4, rect.y, rect.width//2, rect.height//4))
    
    def _draw_enhanced_vehicle(self, surface, vehicle, center, cell_size):
        """Draw enhanced 3D vehicle with details"""
        vehicle_colors = {
            VehicleType.VAN: (30, 100, 200),      # Deep blue
            VehicleType.MOTORCYCLE: (50, 150, 50), # Forest green
            VehicleType.TRUCK: (180, 50, 50)       # Deep red
        }
        
        color = vehicle_colors[vehicle.vehicle_type]
        
        if vehicle.vehicle_type == VehicleType.VAN:
            # Enhanced van with 3D effect
            van_rect = pygame.Rect(center[0] - cell_size//3, center[1] - cell_size//3, 
                                 2*cell_size//3, 2*cell_size//3)
            self._draw_3d_rect(surface, color, van_rect)
            
            # Add van details
            # Front windshield
            windshield = pygame.Rect(van_rect.x + 2, van_rect.y + 2, 
                                   van_rect.width - 4, van_rect.height//3)
            pygame.draw.rect(surface, (200, 220, 255), windshield)
            
            # Side windows
            side_window = pygame.Rect(van_rect.x + 2, van_rect.y + van_rect.height//2, 
                                    van_rect.width - 4, van_rect.height//4)
            pygame.draw.rect(surface, (180, 200, 240), side_window)
            
        elif vehicle.vehicle_type == VehicleType.MOTORCYCLE:
            # Enhanced motorcycle with 3D effect
            self._draw_3d_circle(surface, color, center, cell_size // 3)
            
            # Add motorcycle details
            # Handlebars
            pygame.draw.line(surface, (100, 100, 100),
                           (center[0] - cell_size//4, center[1] - cell_size//6),
                           (center[0] + cell_size//4, center[1] - cell_size//6), 3)
            
            # Wheels
            wheel_color = (50, 50, 50)
            pygame.draw.circle(surface, wheel_color, 
                             (center[0] - cell_size//5, center[1] + cell_size//4), cell_size//8)
            pygame.draw.circle(surface, wheel_color,
                             (center[0] + cell_size//5, center[1] + cell_size//4), cell_size//8)
            
        else:  # TRUCK
            # Enhanced truck with 3D effect
            truck_rect = pygame.Rect(center[0] - cell_size//2, center[1] - cell_size//3,
                                   cell_size, 2*cell_size//3)
            self._draw_3d_rect(surface, color, truck_rect)
            
            # Truck cab
            cab_rect = pygame.Rect(truck_rect.x, truck_rect.y, 
                                 truck_rect.width//3, truck_rect.height)
            self._draw_3d_rect(surface, tuple(c + 20 for c in color), cab_rect)
            
            # Windshield
            windshield = pygame.Rect(cab_rect.x + 2, cab_rect.y + 2,
                                   cab_rect.width - 4, cab_rect.height//2)
            pygame.draw.rect(surface, (200, 220, 255), windshield)
        
        # Enhanced fuel indicator with 3D effect
        fuel_ratio = vehicle.fuel / vehicle.max_fuel
        fuel_bg_rect = pygame.Rect(center[0] - cell_size//3, center[1] + cell_size//2 + 2,
                                 2*cell_size//3, 6)
        pygame.draw.rect(surface, (100, 100, 100), fuel_bg_rect)
        
        fuel_color = (255, 50, 50) if fuel_ratio < 0.3 else (255, 200, 50) if fuel_ratio < 0.6 else (50, 255, 50)
        fuel_rect = pygame.Rect(fuel_bg_rect.x + 1, fuel_bg_rect.y + 1,
                              int((fuel_bg_rect.width - 2) * fuel_ratio), fuel_bg_rect.height - 2)
        pygame.draw.rect(surface, fuel_color, fuel_rect)
        
        # Fuel indicator border
        pygame.draw.rect(surface, (50, 50, 50), fuel_bg_rect, 1)
    
    def _draw_enhanced_delivery_point(self, surface, delivery, pickup_center, delivery_center, cell_size):
        """Draw enhanced delivery points with 3D effects"""
        urgency_colors = {
            UrgencyLevel.STANDARD: (255, 255, 255),
            UrgencyLevel.NORMAL: (255, 220, 100),
            UrgencyLevel.URGENT: (255, 150, 100),
            UrgencyLevel.CRITICAL: (255, 100, 100)
        }
        
        color = urgency_colors[delivery.urgency]
        
        # Pickup point (3D circle with glow)
        glow_radius = cell_size // 3
        for i in range(glow_radius, cell_size//4, -1):
            alpha = int(50 * (glow_radius - i) / (glow_radius - cell_size//4))
            glow_color = (*color, alpha)
            # Create surface for alpha blending
            glow_surface = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (i, i), i)
            surface.blit(glow_surface, (pickup_center[0] - i, pickup_center[1] - i))
        
        self._draw_3d_circle(surface, color, pickup_center, cell_size // 4)
        pygame.draw.circle(surface, (50, 50, 50), pickup_center, cell_size // 4, 2)
        
        # Delivery location (3D diamond with shadow)
        diamond_size = cell_size // 4
        diamond_points = [
            (delivery_center[0], delivery_center[1] - diamond_size),
            (delivery_center[0] + diamond_size, delivery_center[1]),
            (delivery_center[0], delivery_center[1] + diamond_size),
            (delivery_center[0] - diamond_size, delivery_center[1])
        ]
        
        # Shadow
        shadow_points = [(p[0] + 2, p[1] + 2) for p in diamond_points]
        pygame.draw.polygon(surface, (100, 100, 100), shadow_points)
        
        # Main diamond with gradient effect
        pygame.draw.polygon(surface, color, diamond_points)
        
        # Highlight
        highlight_points = [
            diamond_points[0],
            ((diamond_points[0][0] + diamond_points[1][0]) // 2, 
             (diamond_points[0][1] + diamond_points[1][1]) // 2),
            delivery_center,
            ((diamond_points[0][0] + diamond_points[3][0]) // 2,
             (diamond_points[0][1] + diamond_points[3][1]) // 2)
        ]
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.polygon(surface, highlight_color, highlight_points)
        
        pygame.draw.polygon(surface, (50, 50, 50), diamond_points, 2)
    
    def render(self):
        """Render the environment with enhanced visuals"""
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Fleet Management Environment - Urban Logistics Simulator")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface(self.window_size)
        # Create gradient background
        self._draw_gradient_background(canvas)
        
        # Calculate cell size for grid
        grid_width = 600
        cell_size = grid_width // self.grid_size
        grid_offset_x = 50
        grid_offset_y = 50
        
        # Draw customer zones with enhanced 3D buildings
        zone_colors = {
            CustomerZone.RESIDENTIAL: (120, 180, 120),  # Forest green
            CustomerZone.COMMERCIAL: (180, 180, 80),    # Golden yellow
            CustomerZone.INDUSTRIAL: (160, 100, 100),   # Industrial red
            CustomerZone.HOSPITAL: (100, 140, 200)      # Medical blue
        }
        
        zone_types = {
            CustomerZone.RESIDENTIAL: "residential",
            CustomerZone.COMMERCIAL: "commercial", 
            CustomerZone.INDUSTRIAL: "industrial",
            CustomerZone.HOSPITAL: "commercial"
        }
        
        for zone, positions in self.customer_zones.items():
            color = zone_colors[zone]
            building_type = zone_types[zone]
            for pos in positions:
                rect = pygame.Rect(
                    grid_offset_x + pos[0] * cell_size + 1,
                    grid_offset_y + pos[1] * cell_size + 1,
                    cell_size - 2, cell_size - 2
                )
                self._draw_building(canvas, rect, color, building_type)
        
        # Draw subtle grid lines (roads)
        for i in range(self.grid_size + 1):
            # Vertical roads
            pygame.draw.line(canvas, (180, 180, 180),
                           (grid_offset_x + i * cell_size, grid_offset_y),
                           (grid_offset_x + i * cell_size, grid_offset_y + grid_width), 1)
            # Horizontal roads
            pygame.draw.line(canvas, (180, 180, 180),
                           (grid_offset_x, grid_offset_y + i * cell_size),
                           (grid_offset_x + grid_width, grid_offset_y + i * cell_size), 1)
        
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
        
        # Draw enhanced fuel stations
        for station in self.fuel_stations:
            center = (
                grid_offset_x + station[0] * cell_size + cell_size // 2,
                grid_offset_y + station[1] * cell_size + cell_size // 2
            )
            
            # Station base (3D effect)
            station_rect = pygame.Rect(center[0] - cell_size//3, center[1] - cell_size//3,
                                     2*cell_size//3, 2*cell_size//3)
            self._draw_3d_rect(canvas, (40, 120, 40), station_rect)
            
            # Fuel pump
            pump_rect = pygame.Rect(center[0] - cell_size//6, center[1] - cell_size//6,
                                  cell_size//3, cell_size//3)
            self._draw_3d_rect(canvas, (200, 200, 200), pump_rect)
            
            # Fuel symbol
            pygame.draw.circle(canvas, (255, 255, 255), center, cell_size//8)
            pygame.draw.circle(canvas, (50, 150, 50), center, cell_size//10)
        
        # Draw enhanced depot
        depot_rect = pygame.Rect(
            grid_offset_x + self.depot_position[0] * cell_size + 1,
            grid_offset_y + self.depot_position[1] * cell_size + 1,
            cell_size - 2, cell_size - 2
        )
        self._draw_3d_rect(canvas, (80, 80, 120), depot_rect)
        
        # Add depot details
        # Loading dock
        dock_rect = pygame.Rect(depot_rect.x + 2, depot_rect.y + depot_rect.height//2,
                              depot_rect.width - 4, depot_rect.height//4)
        pygame.draw.rect(canvas, (60, 60, 100), dock_rect)
        
        # Depot sign
        sign_rect = pygame.Rect(depot_rect.x + depot_rect.width//4, depot_rect.y + 2,
                              depot_rect.width//2, depot_rect.height//4)
        pygame.draw.rect(canvas, (255, 255, 255), sign_rect)
        pygame.draw.rect(canvas, (0, 0, 0), sign_rect, 1)
        
        # Draw enhanced delivery requests
        for delivery in self.delivery_requests:
            if not delivery.completed:
                # Pickup location (circle)
                pickup_center = (
                    grid_offset_x + delivery.pickup_location[0] * cell_size + cell_size // 2,
                    grid_offset_y + delivery.pickup_location[1] * cell_size + cell_size // 2
                )
                
                # Delivery location (diamond)
                delivery_center = (
                    grid_offset_x + delivery.delivery_location[0] * cell_size + cell_size // 2,
                    grid_offset_y + delivery.delivery_location[1] * cell_size + cell_size // 2
                )
                
                self._draw_enhanced_delivery_point(canvas, delivery, pickup_center, delivery_center, cell_size)
        
        # Draw enhanced vehicles
        for i, vehicle in enumerate(self.vehicles):
            center = (
                grid_offset_x + vehicle.position[0] * cell_size + cell_size // 2,
                grid_offset_y + vehicle.position[1] * cell_size + cell_size // 2
            )
            
            self._draw_enhanced_vehicle(canvas, vehicle, center, cell_size)
        
        # Draw enhanced metrics panel
        panel_x = grid_width + 80
        panel_y = 30
        panel_width = 300
        panel_height = 500
        
        # Panel background with 3D effect
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        self._draw_3d_rect(canvas, (240, 240, 250), panel_rect)
        
        # Title
        title_font = pygame.font.Font(None, 28)
        title_text = title_font.render("Fleet Management", True, (50, 50, 100))
        canvas.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # Separator line
        pygame.draw.line(canvas, (150, 150, 150), 
                        (panel_x + 10, panel_y + 40), 
                        (panel_x + panel_width - 10, panel_y + 40), 2)
        
        # Metrics
        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 18)
        
        y_offset = 55
        
        # Progress bar for timestep
        progress_ratio = self.timestep / self.max_timesteps
        progress_rect = pygame.Rect(panel_x + 10, panel_y + y_offset, panel_width - 20, 15)
        pygame.draw.rect(canvas, (200, 200, 200), progress_rect)
        progress_fill = pygame.Rect(panel_x + 10, panel_y + y_offset, 
                                  int((panel_width - 20) * progress_ratio), 15)
        pygame.draw.rect(canvas, (100, 200, 100), progress_fill)
        pygame.draw.rect(canvas, (100, 100, 100), progress_rect, 1)
        
        timestep_text = small_font.render(f"Timestep: {self.timestep}/{self.max_timesteps}", True, (50, 50, 50))
        canvas.blit(timestep_text, (panel_x + 10, panel_y + y_offset + 20))
        
        y_offset += 50
        
        # Key metrics
        metrics = [
            f"Total Reward: {self.total_reward:.1f}",
            f"Deliveries: {self.completed_deliveries}/{len(self.delivery_requests)}",
            f"Weather: {self.weather_effect:.1f}x",
        ]
        
        for metric in metrics:
            text = font.render(metric, True, (50, 50, 50))
            canvas.blit(text, (panel_x + 10, panel_y + y_offset))
            y_offset += 25
        
        y_offset += 10
        
        # Vehicle status section
        vehicle_title = font.render("Vehicle Status:", True, (50, 50, 100))
        canvas.blit(vehicle_title, (panel_x + 10, panel_y + y_offset))
        y_offset += 30
        
        for i, vehicle in enumerate(self.vehicles):
            # Vehicle type with color indicator
            vehicle_color = {
                VehicleType.VAN: (30, 100, 200),
                VehicleType.MOTORCYCLE: (50, 150, 50),
                VehicleType.TRUCK: (180, 50, 50)
            }[vehicle.vehicle_type]
            
            # Color indicator
            color_rect = pygame.Rect(panel_x + 10, panel_y + y_offset, 15, 15)
            pygame.draw.rect(canvas, vehicle_color, color_rect)
            pygame.draw.rect(canvas, (100, 100, 100), color_rect, 1)
            
            # Vehicle name
            vehicle_text = font.render(f"{vehicle.vehicle_type.value.title()}", True, (50, 50, 50))
            canvas.blit(vehicle_text, (panel_x + 30, panel_y + y_offset))
            
            y_offset += 20
            
            # Fuel bar
            fuel_ratio = vehicle.fuel / vehicle.max_fuel
            fuel_bg_rect = pygame.Rect(panel_x + 15, panel_y + y_offset, 150, 8)
            pygame.draw.rect(canvas, (200, 200, 200), fuel_bg_rect)
            
            fuel_color = (255, 50, 50) if fuel_ratio < 0.3 else (255, 200, 50) if fuel_ratio < 0.6 else (50, 255, 50)
            fuel_fill_rect = pygame.Rect(panel_x + 15, panel_y + y_offset, 
                                       int(150 * fuel_ratio), 8)
            pygame.draw.rect(canvas, fuel_color, fuel_fill_rect)
            pygame.draw.rect(canvas, (100, 100, 100), fuel_bg_rect, 1)
            
            fuel_text = small_font.render(f"Fuel: {vehicle.fuel:.1f}/{vehicle.max_fuel}", True, (80, 80, 80))
            canvas.blit(fuel_text, (panel_x + 170, panel_y + y_offset - 2))
            
            y_offset += 15
            
            # Position and cargo
            pos_text = small_font.render(f"Pos: ({vehicle.position[0]}, {vehicle.position[1]})", True, (80, 80, 80))
            canvas.blit(pos_text, (panel_x + 15, panel_y + y_offset))
            
            cargo_text = small_font.render(f"Cargo: {vehicle.cargo_used}/{vehicle.capacity}", True, (80, 80, 80))
            canvas.blit(cargo_text, (panel_x + 150, panel_y + y_offset))
            
            y_offset += 25
        
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