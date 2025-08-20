"""
Traffic Management Environment - A Gymnasium-compatible environment for traffic light control.

This environment simulates a traffic network with multiple intersections where an RL agent
can control traffic lights to optimize traffic flow.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import random
from typing import Tuple, Dict, Any, Optional, List
import pygame

from config import (
    DEFAULT_GRID_SIZE, DEFAULT_NUM_INTERSECTIONS, MAX_VEHICLES, DEFAULT_SPAWN_RATE,
    LIGHT_PHASES, MAX_TIMESTEPS, INTERSECTION_SPACING, REWARD_VEHICLE_PASSED,
    PENALTY_WAITING_TIME, PENALTY_QUEUE_LENGTH, REWARD_SMOOTH_FLOW,
    PENALTY_EMERGENCY_STOP, MAX_QUEUE_LENGTH, OBSERVATION_RADIUS,
    WINDOW_WIDTH, WINDOW_HEIGHT, COLORS
)
from utils import (
    Vehicle, Intersection, Direction, VehicleType, TrafficLight,
    generate_vehicle_route, get_neighboring_intersections,
    calculate_intersection_position, get_direction_between_intersections,
    calculate_traffic_metrics
)


class TrafficManagementEnv(gym.Env):
    """
    A custom Gymnasium environment for traffic management simulation.
    
    The environment simulates a traffic network with controllable traffic lights.
    The agent's goal is to optimize traffic flow by controlling light phases.
    
    Action Space:
        - MultiDiscrete: One action per controlled intersection
        - Each action: 0 (maintain), 1 (switch to NS_GREEN), 2 (switch to EW_GREEN)
    
    Observation Space:
        - Box: Flattened array containing:
            * Traffic light states for each intersection
            * Queue lengths in each direction for each intersection
            * Waiting times for each intersection
            * Vehicle counts and flow rates
    
    Reward:
        - Positive reward for vehicles passing through intersections
        - Negative penalty for vehicle waiting time
        - Negative penalty for long queues
        - Bonus for smooth traffic flow
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
                 num_intersections: int = DEFAULT_NUM_INTERSECTIONS,
                 max_vehicles: int = MAX_VEHICLES,
                 spawn_rate: float = DEFAULT_SPAWN_RATE,
                 render_mode: Optional[str] = None):
        """
        Initialize the Traffic Management Environment.
        
        Args:
            grid_size: Tuple of (rows, cols) for the intersection grid
            num_intersections: Number of intersections to control
            max_vehicles: Maximum number of vehicles in the simulation
            spawn_rate: Probability of spawning a vehicle per timestep
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_intersections = min(num_intersections, grid_size[0] * grid_size[1])
        self.max_vehicles = max_vehicles
        self.spawn_rate = spawn_rate
        self.render_mode = render_mode
        
        # Initialize intersections
        self.intersections: List[Intersection] = []
        self._create_intersections()
        
        # Vehicle management
        self.vehicles: List[Vehicle] = []
        self.next_vehicle_id = 0
        
        # Simulation state
        self.current_timestep = 0
        self.total_reward = 0.0
        self.episode_metrics = {}
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Rendering
        self.screen = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
    
    def _setup_spaces(self):
        """Set up action and observation spaces."""
        # Action space: 3 actions per intersection (maintain, NS_GREEN, EW_GREEN)
        self.action_space = spaces.MultiDiscrete([3] * self.num_intersections)
        
        # Observation space calculation
        obs_size = (
            self.num_intersections * 4 +  # Traffic light states (4 phases per intersection)
            self.num_intersections * 4 +  # Queue lengths (4 directions per intersection)
            self.num_intersections * 4 +  # Waiting times (4 directions per intersection)
            self.num_intersections * 2 +  # Vehicles passed and total waiting time per intersection
            4  # Global metrics: total vehicles, avg waiting time, avg queue length, throughput
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _create_intersections(self):
        """Create intersections in a grid layout."""
        self.intersections = []
        
        for i in range(self.num_intersections):
            position = calculate_intersection_position(i, self.grid_size, INTERSECTION_SPACING)
            intersection = Intersection(id=i, position=position)
            self.intersections.append(intersection)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[ObsType, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset simulation state
        self.current_timestep = 0
        self.total_reward = 0.0
        self.vehicles = []
        self.next_vehicle_id = 0
        
        # Reset intersections
        for intersection in self.intersections:
            intersection.vehicle_queues = {direction: [] for direction in Direction}
            intersection.vehicles_passed = 0
            intersection.total_waiting_time = 0
            intersection.traffic_light = TrafficLight(intersection.id)
        
        # Generate initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_timestep += 1
        
        # Apply actions to traffic lights
        self._apply_actions(action)
        
        # Update traffic lights
        for intersection in self.intersections:
            intersection.traffic_light.update()
        
        # Spawn new vehicles
        self._spawn_vehicles()
        
        # Update vehicle positions and manage intersections
        self._update_vehicles()
        
        # Process intersections (let vehicles pass)
        self._process_intersections()
        
        # Remove vehicles that have completed their routes
        self._remove_completed_vehicles()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check if episode is done
        terminated = self.current_timestep >= MAX_TIMESTEPS
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_actions(self, actions: np.ndarray):
        """Apply traffic light control actions."""
        for i, action in enumerate(actions):
            if i >= len(self.intersections):
                break
                
            intersection = self.intersections[i]
            traffic_light = intersection.traffic_light
            
            if action == 1:  # Switch to NS_GREEN
                if traffic_light.current_phase != 'NS_GREEN':
                    traffic_light.set_phase('NS_GREEN')
            elif action == 2:  # Switch to EW_GREEN
                if traffic_light.current_phase != 'EW_GREEN':
                    traffic_light.set_phase('EW_GREEN')
            # action == 0: maintain current phase (do nothing)
    
    def _spawn_vehicles(self):
        """Spawn new vehicles at random intersections."""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        if random.random() < self.spawn_rate:
            # Choose random starting intersection
            start_intersection = random.randint(0, self.num_intersections - 1)
            
            # Generate route
            route = generate_vehicle_route(start_intersection, self.grid_size, self.num_intersections)
            
            if len(route) > 1:
                # Create vehicle
                start_pos = self.intersections[start_intersection].position
                direction = get_direction_between_intersections(route[0], route[1], self.grid_size)
                
                vehicle = Vehicle(
                    id=self.next_vehicle_id,
                    position=start_pos,
                    direction=direction,
                    destination=route[-1] if len(route) > 1 else None
                )
                
                # Add to intersection queue
                self.intersections[start_intersection].add_vehicle_to_queue(vehicle)
                self.vehicles.append(vehicle)
                self.next_vehicle_id += 1
    
    def _update_vehicles(self):
        """Update vehicle positions and handle movement between intersections."""
        for vehicle in self.vehicles[:]:  # Copy list to avoid modification during iteration
            # Check if vehicle is at an intersection
            at_intersection = False
            for intersection in self.intersections:
                if vehicle.is_at_intersection(intersection.position):
                    at_intersection = True
                    break
            
            if not at_intersection:
                # Vehicle is traveling between intersections
                vehicle.update_position()
                
                # Check if vehicle has reached the next intersection
                for intersection in self.intersections:
                    if vehicle.is_at_intersection(intersection.position):
                        intersection.add_vehicle_to_queue(vehicle)
                        break
    
    def _process_intersections(self):
        """Process vehicles at each intersection."""
        for intersection in self.intersections:
            proceeding_vehicles = intersection.process_vehicles()
            
            # Update vehicles that can proceed
            for vehicle in proceeding_vehicles:
                # If vehicle has reached its destination, mark for removal
                if (vehicle.destination is not None and 
                    intersection.id == vehicle.destination):
                    vehicle.destination = None  # Mark as completed
    
    def _remove_completed_vehicles(self):
        """Remove vehicles that have completed their routes or left the simulation."""
        self.vehicles = [v for v in self.vehicles if v.destination is not None]
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current timestep."""
        reward = 0.0
        
        # Reward for vehicles passing through intersections
        total_passed = sum(intersection.vehicles_passed for intersection in self.intersections)
        reward += total_passed * REWARD_VEHICLE_PASSED
        
        # Penalty for waiting time
        total_waiting = sum(intersection.total_waiting_time for intersection in self.intersections)
        reward += total_waiting * PENALTY_WAITING_TIME
        
        # Penalty for queue lengths
        total_queue_length = sum(intersection.get_total_queue_length() 
                               for intersection in self.intersections)
        reward += total_queue_length * PENALTY_QUEUE_LENGTH
        
        # Bonus for smooth traffic flow (low variance in queue lengths)
        queue_lengths = [intersection.get_total_queue_length() 
                        for intersection in self.intersections]
        if len(queue_lengths) > 1:
            queue_variance = np.var(queue_lengths)
            reward += REWARD_SMOOTH_FLOW / (1 + queue_variance)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        obs = []
        
        # Traffic light states for each intersection
        for intersection in self.intersections:
            light_state = [0, 0, 0, 0]  # [NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW]
            phase = intersection.traffic_light.current_phase
            if phase == 'NS_GREEN':
                light_state[0] = 1
            elif phase == 'NS_YELLOW':
                light_state[1] = 1
            elif phase == 'EW_GREEN':
                light_state[2] = 1
            elif phase == 'EW_YELLOW':
                light_state[3] = 1
            obs.extend(light_state)
        
        # Queue lengths in each direction for each intersection
        for intersection in self.intersections:
            queue_lengths = intersection.get_queue_lengths()
            obs.extend([min(queue_lengths[direction], MAX_QUEUE_LENGTH) 
                       for direction in Direction])
        
        # Waiting times for each intersection (normalized)
        for intersection in self.intersections:
            avg_waiting_times = []
            for direction in Direction:
                queue = intersection.vehicle_queues[direction]
                if queue:
                    avg_waiting = sum(v.waiting_time for v in queue) / len(queue)
                else:
                    avg_waiting = 0
                avg_waiting_times.append(min(avg_waiting, 100))  # Cap at 100
            obs.extend(avg_waiting_times)
        
        # Vehicles passed and total waiting time per intersection
        for intersection in self.intersections:
            obs.append(intersection.vehicles_passed)
            obs.append(min(intersection.total_waiting_time, 1000))  # Cap at 1000
        
        # Global metrics
        metrics = calculate_traffic_metrics(self.intersections)
        obs.extend([
            len(self.vehicles),
            min(metrics['average_waiting_time'], 100),
            min(metrics['average_queue_length'], 50),
            metrics['throughput']
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        metrics = calculate_traffic_metrics(self.intersections)
        
        return {
            'timestep': self.current_timestep,
            'num_vehicles': len(self.vehicles),
            'total_reward': self.total_reward,
            'metrics': metrics,
            'intersection_states': [
                {
                    'id': intersection.id,
                    'light_phase': intersection.traffic_light.current_phase,
                    'queue_lengths': intersection.get_queue_lengths(),
                    'vehicles_passed': intersection.vehicles_passed,
                    'total_waiting_time': intersection.total_waiting_time
                }
                for intersection in self.intersections
            ]
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.screen.fill(COLORS['BACKGROUND'])
        
        # Draw intersections and traffic lights
        self._draw_intersections()
        
        # Draw vehicles
        self._draw_vehicles()
        
        # Draw UI information
        self._draw_ui()
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self):
        """Render as RGB array."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HIDDEN)
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        self.screen.fill(COLORS['BACKGROUND'])
        self._draw_intersections()
        self._draw_vehicles()
        self._draw_ui()
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def _draw_intersections(self):
        """Draw intersections and traffic lights."""
        for intersection in self.intersections:
            x, y = intersection.position
            
            # Scale position to screen coordinates
            screen_x = int(x * WINDOW_WIDTH / (self.grid_size[1] * INTERSECTION_SPACING))
            screen_y = int(y * WINDOW_HEIGHT / (self.grid_size[0] * INTERSECTION_SPACING))
            
            # Draw intersection
            pygame.draw.circle(self.screen, COLORS['INTERSECTION'], 
                             (screen_x, screen_y), 30)
            
            # Draw traffic light
            light_color = COLORS['RED_LIGHT']
            if intersection.traffic_light.current_phase == 'NS_GREEN':
                light_color = COLORS['GREEN_LIGHT']
            elif intersection.traffic_light.current_phase == 'EW_GREEN':
                light_color = COLORS['GREEN_LIGHT']
            elif 'YELLOW' in intersection.traffic_light.current_phase:
                light_color = COLORS['YELLOW_LIGHT']
            
            pygame.draw.circle(self.screen, light_color, 
                             (screen_x, screen_y), 15)
            
            # Draw queue lengths
            queue_lengths = intersection.get_queue_lengths()
            total_queue = sum(queue_lengths.values())
            if total_queue > 0:
                font = pygame.font.Font(None, 24)
                text = font.render(str(total_queue), True, COLORS['TEXT'])
                self.screen.blit(text, (screen_x - 10, screen_y - 50))
    
    def _draw_vehicles(self):
        """Draw vehicles in the simulation."""
        for vehicle in self.vehicles:
            x, y = vehicle.position
            
            # Scale position to screen coordinates
            screen_x = int(x * WINDOW_WIDTH / (self.grid_size[1] * INTERSECTION_SPACING))
            screen_y = int(y * WINDOW_HEIGHT / (self.grid_size[0] * INTERSECTION_SPACING))
            
            # Choose color based on vehicle type
            color = COLORS['VEHICLE']
            if vehicle.vehicle_type == VehicleType.EMERGENCY:
                color = COLORS['EMERGENCY_VEHICLE']
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 4)
    
    def _draw_ui(self):
        """Draw UI information."""
        font = pygame.font.Font(None, 36)
        
        # Draw timestep
        timestep_text = font.render(f"Timestep: {self.current_timestep}", 
                                  True, COLORS['TEXT'])
        self.screen.blit(timestep_text, (10, 10))
        
        # Draw vehicle count
        vehicle_text = font.render(f"Vehicles: {len(self.vehicles)}", 
                                 True, COLORS['TEXT'])
        self.screen.blit(vehicle_text, (10, 50))
        
        # Draw total reward
        reward_text = font.render(f"Reward: {self.total_reward:.2f}", 
                                True, COLORS['TEXT'])
        self.screen.blit(reward_text, (10, 90))
    
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None