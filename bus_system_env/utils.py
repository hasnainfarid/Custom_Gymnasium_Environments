"""
Utility functions for the Bus System Environment.
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from config import NUM_STOPS, MAX_PASSENGERS, TRAVEL_TIME_BETWEEN_STOPS


class Passenger:
    """Represents a passenger with source and destination stops."""
    
    def __init__(self, source: int, destination: int):
        self.source = source
        self.destination = destination
    
    def __repr__(self):
        return f"Passenger({self.source} -> {self.destination})"


def generate_passengers() -> Dict[int, List[Passenger]]:
    """
    Generate random passengers distributed across stops.
    
    Returns:
        Dictionary mapping stop index to list of waiting passengers
    """
    passengers = {i: [] for i in range(NUM_STOPS)}
    
    # Generate random number of passengers (up to MAX_PASSENGERS)
    num_passengers = random.randint(50, MAX_PASSENGERS)
    
    for _ in range(num_passengers):
        # Random source stop
        source = random.randint(0, NUM_STOPS - 1)
        
        # Random destination (different from source)
        destination = random.randint(0, NUM_STOPS - 1)
        while destination == source:
            destination = random.randint(0, NUM_STOPS - 1)
        
        # Add passenger to source stop
        passengers[source].append(Passenger(source, destination))
    
    return passengers


def get_destination_distribution(passengers: List[Passenger]) -> np.ndarray:
    """
    Get the distribution of destinations for a list of passengers.
    
    Args:
        passengers: List of passengers
        
    Returns:
        Array of length NUM_STOPS with count of passengers going to each stop
    """
    distribution = np.zeros(NUM_STOPS, dtype=np.int32)
    for passenger in passengers:
        distribution[passenger.destination] += 1
    return distribution


def calculate_travel_time(from_stop: int, to_stop: int) -> int:
    """
    Calculate travel time between two stops.
    
    Args:
        from_stop: Source stop index
        to_stop: Destination stop index
        
    Returns:
        Travel time in timesteps
    """
    # For circular route, calculate shortest path
    distance = abs(to_stop - from_stop)
    # Handle circular distance
    distance = min(distance, NUM_STOPS - distance)
    return distance * TRAVEL_TIME_BETWEEN_STOPS


def get_next_stop(current_stop: int) -> int:
    """
    Get the next stop in the circular route.
    
    Args:
        current_stop: Current stop index
        
    Returns:
        Next stop index
    """
    return (current_stop + 1) % NUM_STOPS


def get_previous_stop(current_stop: int) -> int:
    """
    Get the previous stop in the circular route.
    
    Args:
        current_stop: Current stop index
        
    Returns:
        Previous stop index
    """
    return (current_stop - 1) % NUM_STOPS


def format_time(timesteps: int) -> str:
    """
    Format timesteps into a readable time string.
    
    Args:
        timesteps: Number of timesteps
        
    Returns:
        Formatted time string
    """
    minutes = timesteps // 10  # Assuming 10 timesteps = 1 minute
    seconds = (timesteps % 10) * 6  # 6 seconds per timestep
    return f"{minutes:02d}:{seconds:02d}"


def validate_stop_index(stop: int) -> bool:
    """
    Validate if a stop index is within valid range.
    
    Args:
        stop: Stop index to validate
        
    Returns:
        True if valid, False otherwise
    """
    return 0 <= stop < NUM_STOPS


def count_total_passengers(passengers_dict: Dict[int, List[Passenger]]) -> int:
    """
    Count total number of passengers across all stops.
    
    Args:
        passengers_dict: Dictionary mapping stops to passenger lists
        
    Returns:
        Total number of passengers
    """
    return sum(len(passengers) for passengers in passengers_dict.values()) 