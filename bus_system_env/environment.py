"""
Bus System Environment - A Gymnasium-compatible environment for urban bus system simulation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import random
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from config import (
    NUM_STOPS, NUM_BUSES, BUS_CAPACITY, MAX_TIMESTEPS, MAX_DWELL_TIME,
    REWARD_PER_DELIVERY, PENALTY_PER_WAITING_PASSENGER, PENALTY_PER_ONBOARD_PASSENGER
)
from utils import Passenger, generate_passengers, get_destination_distribution, get_next_stop


@dataclass
class BusState:
    """Represents the state of a single bus."""
    current_stop: int = 0
    is_stopped: bool = True
    remaining_time: int = 0  # Travel time or dwell time remaining
    onboard_passengers: List[Passenger] = None
    dwell_time: int = 0  # Current dwell time setting
    
    def __post_init__(self):
        if self.onboard_passengers is None:
            self.onboard_passengers = []
    
    @property
    def capacity_remaining(self) -> int:
        """Get remaining capacity of the bus."""
        return BUS_CAPACITY - len(self.onboard_passengers)
    
    @property
    def is_full(self) -> bool:
        """Check if bus is at full capacity."""
        return len(self.onboard_passengers) >= BUS_CAPACITY


class BusSystemEnv(gym.Env):
    """
    A custom Gymnasium environment simulating an urban bus system.
    
    Features:
    - 4 buses operating on a circular route with 4 stops
    - Dynamic passenger generation and management
    - Centralized dwell time control
    - Realistic passenger boarding and alighting
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, max_timesteps: int = MAX_TIMESTEPS, 
                 enable_visualization: bool = False, demo_speed: float = 1.0):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.enable_visualization = enable_visualization
        self.demo_speed = demo_speed
        
        # Environment state
        self.buses = [BusState() for i in range(NUM_BUSES)]
        self.waiting_passengers = {i: [] for i in range(NUM_STOPS)}
        self.total_delivered = 0
        self.total_waiting = 0
        self.total_onboard = 0
        
        # Initialize buses at different stops
        for i, bus in enumerate(self.buses):
            bus.current_stop = i % NUM_STOPS
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space: dwell times for each bus (0 to MAX_DWELL_TIME)
        self.action_space = spaces.MultiDiscrete([MAX_DWELL_TIME + 1] * NUM_BUSES)
        
        # Initialize renderer if needed
        self.renderer = None
        if self.enable_visualization:
            try:
                from pygame_visualizer import BusSystemVisualizer
                self.renderer = BusSystemVisualizer(demo_mode=True, demo_speed=self.demo_speed)
            except ImportError:
                print("Warning: Pygame not available. Running without visualization.")
                self.enable_visualization = False
    
    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space for the environment."""
        return spaces.Dict({
            # Bus information
            "bus_stops": spaces.MultiDiscrete([NUM_STOPS] * NUM_BUSES),  # Current stop for each bus
            "bus_states": spaces.MultiBinary(NUM_BUSES),  # 1 if stopped, 0 if en route
            "bus_remaining_times": spaces.Box(
                low=0, high=50, shape=(NUM_BUSES,), dtype=np.int32
            ),  # Remaining travel/dwell time
            "bus_capacities": spaces.Box(
                low=0, high=BUS_CAPACITY, shape=(NUM_BUSES,), dtype=np.int32
            ),  # Remaining capacity for each bus
            "bus_passenger_destinations": spaces.Box(
                low=0, high=50, shape=(NUM_BUSES, NUM_STOPS), dtype=np.int32
            ),  # Destination distribution for onboard passengers
            
            # Stop information
            "stop_waiting_counts": spaces.Box(
                low=0, high=100, shape=(NUM_STOPS,), dtype=np.int32
            ),  # Number of waiting passengers at each stop
            "stop_destination_distributions": spaces.Box(
                low=0, high=50, shape=(NUM_STOPS, NUM_STOPS), dtype=np.int32
            ),  # Destination distribution for waiting passengers
            
            # Global information
            "timestep": spaces.Discrete(self.max_timesteps),
            "total_delivered": spaces.Discrete(1000),
            "total_waiting": spaces.Discrete(1000),
            "total_onboard": spaces.Discrete(1000)
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_timestep = 0
        self.total_delivered = 0
        self.total_waiting = 0
        self.total_onboard = 0
        
        # Reset buses
        for i, bus in enumerate(self.buses):
            bus.current_stop = i % NUM_STOPS
            bus.is_stopped = True
            bus.remaining_time = 0
            bus.onboard_passengers = []
            bus.dwell_time = 0
        
        # Generate new passengers
        self.waiting_passengers = generate_passengers()
        self._update_passenger_counts()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        reward = 0.0
        terminated = False
        truncated = False
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be array of {NUM_BUSES} integers 0-{MAX_DWELL_TIME}")
        
        # Store the agent's dwell_time choices for buses that will arrive at a stop this step
        self._pending_dwell_times = list(action)
        
        # Update bus states and process passenger movements
        reward = self._update_buses()
        
        # Process passenger boarding and alighting for all stopped buses (regardless of remaining_time)
        reward += self._process_passenger_movements()
        
        # Increment timestep
        self.current_timestep += 1
        
        # Check termination conditions
        if self.current_timestep >= self.max_timesteps:
            truncated = True
        
        # Update passenger counts
        self._update_passenger_counts()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _update_buses(self) -> float:
        """Update bus states and return reward for time penalties."""
        reward = 0.0
        
        for bus_id, bus in enumerate(self.buses):
            if bus.remaining_time > 0:
                # Bus is traveling or dwelling
                bus.remaining_time -= 1
                if bus.remaining_time == 0:
                    # Bus has arrived at destination or finished dwelling
                    if bus.is_stopped:
                        # Finished dwelling, start traveling to next stop
                        bus.is_stopped = False
                        bus.current_stop = get_next_stop(bus.current_stop)  # Move to next stop
                        bus.remaining_time = 5  # Travel time between stops
                    else:
                        # Arrived at stop, start dwelling if needed
                        bus.is_stopped = True
                        # Only set dwell time if agent specified > 0
                        # Use the agent's dwell_time action for this bus
                        if hasattr(self, '_pending_dwell_times'):
                            bus.dwell_time = self._pending_dwell_times[bus_id]
                        if bus.dwell_time > 0:
                            bus.remaining_time = bus.dwell_time
                        # If dwell_time is 0, bus can immediately process passengers

            # FIXED LOGIC: Allow buses to move when they're full or when they've been at a stop long enough
            # This prevents deadlock and ensures passenger delivery
            if (bus.is_stopped and bus.remaining_time == 0):
                # Bus can move if:
                # 1. Bus is full (need to deliver passengers)
                # 2. No waiting passengers at current stop
                # 3. Bus has been at stop for a while (to prevent immediate jumping)
                if (bus.is_full or 
                    len(self.waiting_passengers[bus.current_stop]) == 0 or
                    self.current_timestep % 3 == 0):  # Move every 3 steps if stuck
                    
                    bus.is_stopped = False
                    bus.current_stop = get_next_stop(bus.current_stop)
                    bus.remaining_time = 5  # Travel time between stops

            # Apply time penalties
            if bus.is_stopped:
                # Penalty for onboard passengers while stopped
                reward += len(bus.onboard_passengers) * PENALTY_PER_ONBOARD_PASSENGER
            else:
                # Penalty for onboard passengers while traveling
                reward += len(bus.onboard_passengers) * PENALTY_PER_ONBOARD_PASSENGER
        
        return reward
    
    def _process_passenger_movements(self) -> float:
        """Process passenger boarding and alighting, return reward for deliveries."""
        reward = 0.0
        
        for bus in self.buses:
            # Always process if bus is stopped (regardless of remaining_time)
            if not bus.is_stopped:
                continue  # Bus is traveling
            
            current_stop = bus.current_stop
            
            # Process alighting (passengers getting off) - ALWAYS when stopped
            alighting_passengers = []
            for passenger in bus.onboard_passengers:
                if passenger.destination == current_stop:
                    alighting_passengers.append(passenger)
                    reward += REWARD_PER_DELIVERY
                    self.total_delivered += 1
            
            # Remove alighting passengers
            for passenger in alighting_passengers:
                bus.onboard_passengers.remove(passenger)
            
            # Process boarding (passengers getting on) - when stopped, regardless of dwell time
            if not bus.is_full and self.waiting_passengers[current_stop]:
                # Sort passengers by destination to optimize route
                self.waiting_passengers[current_stop].sort(
                    key=lambda p: self._calculate_boarding_priority(p, bus)
                )
                
                # Board passengers until bus is full or no more waiting
                while (not bus.is_full and 
                       self.waiting_passengers[current_stop]):
                    
                    passenger = self.waiting_passengers[current_stop].pop(0)
                    bus.onboard_passengers.append(passenger)
        
        return reward
    
    def _calculate_boarding_priority(self, passenger: Passenger, bus: BusState) -> int:
        """Calculate boarding priority for a passenger based on bus route."""
        # Higher priority for passengers going in the direction the bus is traveling
        bus_next_stop = get_next_stop(bus.current_stop)
        
        # Calculate shortest path to destination
        direct_distance = abs(passenger.destination - bus.current_stop)
        circular_distance = NUM_STOPS - direct_distance
        shortest_distance = min(direct_distance, circular_distance)
        
        # Check if passenger destination is in the direction the bus is going
        if bus_next_stop == passenger.destination:
            return 0  # Highest priority
        elif shortest_distance <= 2:  # Close destinations get higher priority
            return shortest_distance
        else:
            return shortest_distance + 10  # Lower priority for far destinations
    
    def _update_passenger_counts(self):
        """Update total passenger counts."""
        self.total_waiting = sum(len(passengers) for passengers in self.waiting_passengers.values())
        self.total_onboard = sum(len(bus.onboard_passengers) for bus in self.buses)
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        # Bus information
        bus_stops = np.array([bus.current_stop for bus in self.buses], dtype=np.int32)
        bus_states = np.array([1 if bus.is_stopped else 0 for bus in self.buses], dtype=np.int8)
        bus_remaining_times = np.array([bus.remaining_time for bus in self.buses], dtype=np.int32)
        bus_capacities = np.array([bus.capacity_remaining for bus in self.buses], dtype=np.int32)
        
        # Bus passenger destination distributions
        bus_passenger_destinations = np.zeros((NUM_BUSES, NUM_STOPS), dtype=np.int32)
        for i, bus in enumerate(self.buses):
            bus_passenger_destinations[i] = get_destination_distribution(bus.onboard_passengers)
        
        # Stop information
        stop_waiting_counts = np.array([
            len(self.waiting_passengers[i]) for i in range(NUM_STOPS)
        ], dtype=np.int32)
        
        stop_destination_distributions = np.zeros((NUM_STOPS, NUM_STOPS), dtype=np.int32)
        for i in range(NUM_STOPS):
            stop_destination_distributions[i] = get_destination_distribution(
                self.waiting_passengers[i]
            )
        
        return {
            "bus_stops": bus_stops,
            "bus_states": bus_states,
            "bus_remaining_times": bus_remaining_times,
            "bus_capacities": bus_capacities,
            "bus_passenger_destinations": bus_passenger_destinations,
            "stop_waiting_counts": stop_waiting_counts,
            "stop_destination_distributions": stop_destination_distributions,
            "timestep": self.current_timestep,
            "total_delivered": self.total_delivered,
            "total_waiting": self.total_waiting,
            "total_onboard": self.total_onboard
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "timestep": self.current_timestep,
            "total_delivered": self.total_delivered,
            "total_waiting": self.total_waiting,
            "total_onboard": self.total_onboard,
            "bus_positions": [bus.current_stop for bus in self.buses],
            "bus_states": ["stopped" if bus.is_stopped else "traveling" for bus in self.buses],
            "bus_capacities": [bus.capacity_remaining for bus in self.buses],
            "stop_waiting": [len(passengers) for passengers in self.waiting_passengers.values()]
        }
    
    def render(self):
        """Render the environment."""
        if self.enable_visualization and self.renderer is not None:
            self.renderer.render(self)
        elif self.render_mode == "rgb_array":
            # Return a simple RGB array representation
            return np.zeros((600, 800, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
    
    def get_state_summary(self) -> str:
        """Get a human-readable summary of the current state."""
        summary = f"Timestep: {self.current_timestep}\n"
        summary += f"Total Delivered: {self.total_delivered}\n"
        summary += f"Total Waiting: {self.total_waiting}\n"
        summary += f"Total Onboard: {self.total_onboard}\n\n"
        
        summary += "Buses:\n"
        for i, bus in enumerate(self.buses):
            state = "STOPPED" if bus.is_stopped else "TRAVELING"
            summary += f"  Bus {i}: Stop {bus.current_stop}, {state}, "
            summary += f"Capacity: {bus.capacity_remaining}/{BUS_CAPACITY}, "
            summary += f"Passengers: {len(bus.onboard_passengers)}\n"
        
        summary += "\nStops:\n"
        for i in range(NUM_STOPS):
            waiting = len(self.waiting_passengers[i])
            summary += f"  Stop {i}: {waiting} waiting passengers\n"
        
        return summary 