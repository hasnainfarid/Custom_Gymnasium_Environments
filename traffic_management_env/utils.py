"""
Utility classes and functions for the Traffic Management Environment.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import (
    VEHICLE_SPEED, VEHICLE_LENGTH, LIGHT_PHASES, YELLOW_DURATION,
    MIN_PHASE_DURATION, MAX_PHASE_DURATION
)


class Direction(Enum):
    """Vehicle movement directions."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class VehicleType(Enum):
    """Types of vehicles in the simulation."""
    REGULAR = 0
    EMERGENCY = 1
    PUBLIC_TRANSPORT = 2


@dataclass
class Vehicle:
    """Represents a vehicle in the traffic simulation."""
    id: int
    position: Tuple[float, float]  # (x, y) coordinates
    direction: Direction
    speed: float = VEHICLE_SPEED
    vehicle_type: VehicleType = VehicleType.REGULAR
    destination: Optional[Tuple[int, int]] = None  # Target intersection
    waiting_time: int = 0  # Time spent waiting at intersections
    total_travel_time: int = 0  # Total time in simulation
    
    def update_position(self, dt: float = 1.0):
        """Update vehicle position based on current speed and direction."""
        dx, dy = self._get_direction_vector()
        self.position = (
            self.position[0] + dx * self.speed * dt,
            self.position[1] + dy * self.speed * dt
        )
        self.total_travel_time += 1
    
    def _get_direction_vector(self) -> Tuple[float, float]:
        """Get the direction vector for movement."""
        direction_vectors = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0)
        }
        return direction_vectors[self.direction]
    
    def is_at_intersection(self, intersection_pos: Tuple[float, float], 
                          tolerance: float = 5.0) -> bool:
        """Check if vehicle is at a specific intersection."""
        dist = np.sqrt((self.position[0] - intersection_pos[0])**2 + 
                      (self.position[1] - intersection_pos[1])**2)
        return dist <= tolerance


@dataclass
class TrafficLight:
    """Represents a traffic light at an intersection."""
    intersection_id: int
    current_phase: str = 'NS_GREEN'  # Current light phase
    phase_timer: int = 0  # Time remaining in current phase
    phase_duration: int = MIN_PHASE_DURATION
    
    def update(self):
        """Update traffic light state."""
        self.phase_timer -= 1
        if self.phase_timer <= 0:
            self._advance_phase()
    
    def _advance_phase(self):
        """Advance to the next traffic light phase."""
        current_idx = LIGHT_PHASES.index(self.current_phase)
        next_idx = (current_idx + 1) % len(LIGHT_PHASES)
        self.current_phase = LIGHT_PHASES[next_idx]
        
        # Set duration for new phase
        if 'YELLOW' in self.current_phase:
            self.phase_duration = YELLOW_DURATION
        else:
            self.phase_duration = random.randint(MIN_PHASE_DURATION, MAX_PHASE_DURATION)
        
        self.phase_timer = self.phase_duration
    
    def can_pass(self, direction: Direction) -> bool:
        """Check if a vehicle can pass through the intersection."""
        if self.current_phase == 'NS_GREEN':
            return direction in [Direction.NORTH, Direction.SOUTH]
        elif self.current_phase == 'EW_GREEN':
            return direction in [Direction.EAST, Direction.WEST]
        else:  # Yellow phases
            return False  # Vehicles should stop during yellow
    
    def set_phase(self, phase: str, duration: int = None):
        """Manually set traffic light phase (for RL control)."""
        if phase in LIGHT_PHASES:
            self.current_phase = phase
            if duration is None:
                if 'YELLOW' in phase:
                    duration = YELLOW_DURATION
                else:
                    duration = MIN_PHASE_DURATION
            self.phase_duration = duration
            self.phase_timer = duration


@dataclass
class Intersection:
    """Represents a traffic intersection."""
    id: int
    position: Tuple[float, float]  # (x, y) coordinates
    traffic_light: TrafficLight = field(init=False)
    vehicle_queues: Dict[Direction, List[Vehicle]] = field(default_factory=dict)
    vehicles_passed: int = 0  # Count of vehicles that passed through
    total_waiting_time: int = 0  # Cumulative waiting time of all vehicles
    
    def __post_init__(self):
        self.traffic_light = TrafficLight(self.id)
        # Initialize queues for all directions
        for direction in Direction:
            self.vehicle_queues[direction] = []
    
    def add_vehicle_to_queue(self, vehicle: Vehicle):
        """Add a vehicle to the appropriate queue."""
        self.vehicle_queues[vehicle.direction].append(vehicle)
    
    def process_vehicles(self) -> List[Vehicle]:
        """Process vehicles at the intersection and return vehicles that can proceed."""
        proceeding_vehicles = []
        
        for direction in Direction:
            queue = self.vehicle_queues[direction]
            if not queue:
                continue
            
            # Check if vehicles can pass
            if self.traffic_light.can_pass(direction):
                # Allow vehicles to pass
                while queue:
                    vehicle = queue.pop(0)
                    proceeding_vehicles.append(vehicle)
                    self.vehicles_passed += 1
            else:
                # Update waiting time for queued vehicles
                for vehicle in queue:
                    vehicle.waiting_time += 1
                    self.total_waiting_time += 1
        
        return proceeding_vehicles
    
    def get_queue_lengths(self) -> Dict[Direction, int]:
        """Get the length of queues in each direction."""
        return {direction: len(queue) for direction, queue in self.vehicle_queues.items()}
    
    def get_total_queue_length(self) -> int:
        """Get total number of vehicles waiting at this intersection."""
        return sum(len(queue) for queue in self.vehicle_queues.values())


def generate_vehicle_route(start_intersection: int, grid_size: Tuple[int, int],
                          num_intersections: int) -> List[int]:
    """Generate a random route for a vehicle through the traffic network."""
    route = [start_intersection]
    current = start_intersection
    
    # Generate a route of 2-5 intersections
    route_length = random.randint(2, min(5, num_intersections))
    
    for _ in range(route_length - 1):
        # Get neighboring intersections
        neighbors = get_neighboring_intersections(current, grid_size)
        if neighbors:
            next_intersection = random.choice(neighbors)
            route.append(next_intersection)
            current = next_intersection
        else:
            break
    
    return route


def get_neighboring_intersections(intersection_id: int, 
                                grid_size: Tuple[int, int]) -> List[int]:
    """Get neighboring intersections in a grid layout."""
    rows, cols = grid_size
    row = intersection_id // cols
    col = intersection_id % cols
    
    neighbors = []
    
    # Check all four directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # North, South, West, East
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbor_id = new_row * cols + new_col
            neighbors.append(neighbor_id)
    
    return neighbors


def calculate_intersection_position(intersection_id: int, grid_size: Tuple[int, int],
                                  spacing: float = 100.0) -> Tuple[float, float]:
    """Calculate the (x, y) position of an intersection in the grid."""
    rows, cols = grid_size
    row = intersection_id // cols
    col = intersection_id % cols
    
    x = col * spacing
    y = row * spacing
    
    return (x, y)


def get_direction_between_intersections(from_id: int, to_id: int,
                                      grid_size: Tuple[int, int]) -> Direction:
    """Get the direction to travel from one intersection to another."""
    rows, cols = grid_size
    
    from_row, from_col = from_id // cols, from_id % cols
    to_row, to_col = to_id // cols, to_id % cols
    
    if to_row < from_row:
        return Direction.NORTH
    elif to_row > from_row:
        return Direction.SOUTH
    elif to_col < from_col:
        return Direction.WEST
    elif to_col > from_col:
        return Direction.EAST
    else:
        # Same intersection, return random direction
        return random.choice(list(Direction))


def calculate_traffic_metrics(intersections: List[Intersection]) -> Dict[str, float]:
    """Calculate various traffic performance metrics."""
    total_vehicles_passed = sum(intersection.vehicles_passed for intersection in intersections)
    total_waiting_time = sum(intersection.total_waiting_time for intersection in intersections)
    total_queue_length = sum(intersection.get_total_queue_length() for intersection in intersections)
    
    avg_waiting_time = total_waiting_time / max(total_vehicles_passed, 1)
    avg_queue_length = total_queue_length / len(intersections)
    
    return {
        'total_vehicles_passed': total_vehicles_passed,
        'total_waiting_time': total_waiting_time,
        'average_waiting_time': avg_waiting_time,
        'total_queue_length': total_queue_length,
        'average_queue_length': avg_queue_length,
        'throughput': total_vehicles_passed / len(intersections)
    }