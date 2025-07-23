"""
Parking lot management for the Smart Parking Lot RL Environment.
"""

import random
from typing import Dict, List, Optional, Tuple

# Paperspace-compatible imports
try:
    from .config import ZONE_CONFIG, MAX_QUEUE_SIZE
    from .customer import Customer
except ImportError:
    from config import ZONE_CONFIG, MAX_QUEUE_SIZE
    from customer import Customer


class ParkingSpot:
    """
    Represents a single parking spot.
    """
    
    def __init__(self, spot_id: int, zone: str):
        """
        Initialize a parking spot.
        
        Args:
            spot_id: Unique identifier for the spot
            zone: Zone where the spot is located
        """
        self.spot_id = spot_id
        self.zone = zone
        self.is_occupied = False
        self.customer_id = None
        self.remaining_hours = 0
        self.arrival_timestep = None
        
    def assign_customer(self, customer: Customer, current_timestep: int) -> None:
        """
        Assign a customer to this spot.
        
        Args:
            customer: Customer to assign
            current_timestep: Current timestep
        """
        self.is_occupied = True
        self.customer_id = customer.customer_id
        self.remaining_hours = customer.duration
        self.arrival_timestep = current_timestep
        
    def release_spot(self) -> Optional[int]:
        """
        Release the spot and return the customer ID.
        
        Returns:
            Customer ID that was using the spot, or None if empty
        """
        if not self.is_occupied:
            return None
            
        customer_id = self.customer_id
        self.is_occupied = False
        self.customer_id = None
        self.remaining_hours = 0
        self.arrival_timestep = None
        return customer_id
        
    def update_time_minute(self, current_timestep: int) -> bool:
        """
        Update the remaining time for the parked car (minute-based).
        
        Args:
            current_timestep: Current timestep
            
        Returns:
            True if car should leave, False otherwise
        """
        if not self.is_occupied or self.arrival_timestep is None:
            return False
        
        # Calculate hours parked (in minutes, convert to hours)
        minutes_parked = current_timestep - self.arrival_timestep
        hours_parked = minutes_parked / 60.0
        
        # Check if customer should leave (duration exceeded)
        return hours_parked >= self.remaining_hours
        
    def __str__(self) -> str:
        """String representation of parking spot."""
        status = f"Occupied (Customer {self.customer_id}, {self.remaining_hours}h)" if self.is_occupied else "Empty"
        return f"Spot {self.spot_id} ({self.zone}): {status}"


class ParkingLot:
    """
    Manages the parking lot state including spots, zones, and queue.
    """
    
    def __init__(self):
        """Initialize the parking lot."""
        self.spots: Dict[int, ParkingSpot] = {}
        self.zone_spots: Dict[str, List[int]] = {zone: [] for zone in ZONE_CONFIG.keys()}
        self.queue: List[Customer] = [] # Changed from deque to List for Paperspace compatibility
        self.total_spots = 0
        
        # Initialize spots
        spot_id = 0
        for zone, config in ZONE_CONFIG.items():
            for _ in range(config['spots']):
                spot = ParkingSpot(spot_id, zone)
                self.spots[spot_id] = spot
                self.zone_spots[zone].append(spot_id)
                spot_id += 1
                self.total_spots += 1
                
    def get_available_spots(self, zone: str) -> List[int]:
        """
        Get list of available spot IDs in a zone.
        
        Args:
            zone: Zone to check
            
        Returns:
            List of available spot IDs
        """
        available = []
        for spot_id in self.zone_spots[zone]:
            if not self.spots[spot_id].is_occupied:
                available.append(spot_id)
        return available
        
    def get_total_available_spots(self) -> int:
        """
        Get total number of available spots across all zones.
        
        Returns:
            Total number of available spots
        """
        return sum(len(self.get_available_spots(zone)) for zone in ['A', 'B', 'C'])
        
    def get_zone_occupancy(self, zone: str) -> Tuple[int, int]:
        """
        Get occupancy statistics for a zone.
        
        Args:
            zone: Zone to check
            
        Returns:
            Tuple of (occupied_spots, total_spots)
        """
        total_spots = len(self.zone_spots[zone])
        occupied_spots = sum(1 for spot_id in self.zone_spots[zone] 
                           if self.spots[spot_id].is_occupied)
        return occupied_spots, total_spots
        
    def get_overall_occupancy(self) -> Tuple[int, int]:
        """
        Get overall occupancy statistics.
        
        Returns:
            Tuple of (occupied_spots, total_spots)
        """
        occupied_spots = sum(1 for spot in self.spots.values() if spot.is_occupied)
        return occupied_spots, self.total_spots
        
    def assign_customer_to_spot(self, customer: Customer, zone: str, 
                               spot_id: int, price: float, current_timestep: int) -> bool:
        """
        Assign a customer to a specific spot.
        
        Args:
            customer: Customer to assign
            zone: Zone for assignment
            spot_id: Spot ID to assign
            price: Price for the spot
            current_timestep: Current timestep
            
        Returns:
            True if assignment successful, False otherwise
        """
        if spot_id not in self.spots:
            return False
            
        spot = self.spots[spot_id]
        if spot.is_occupied or spot.zone != zone:
            return False
            
        # Assign the spot
        spot.assign_customer(customer, current_timestep)
        customer.assign_spot(zone, spot_id, price)
        
        return True
        
    def add_to_queue(self, customer: Customer) -> bool:
        """
        Add a customer to the queue.
        
        Args:
            customer: Customer to add to queue
            
        Returns:
            True if added successfully, False if queue is full
        """
        if len(self.queue) >= MAX_QUEUE_SIZE:
            return False
            
        self.queue.append(customer)
        customer.add_to_queue(len(self.queue) - 1)
        return True
        
    def get_next_queued_customer(self) -> Optional[Customer]:
        """
        Get the next customer from the queue.
        
        Returns:
            Next customer or None if queue is empty
        """
        if not self.queue:
            return None
            
        customer = self.queue.pop(0) # Changed from popleft to pop(0)
        customer.remove_from_queue()
        
        # Update queue positions for remaining customers
        for i, queued_customer in enumerate(self.queue):
            queued_customer.queue_position = i
            
        return customer
        
    def update_time_minute(self, current_timestep: int) -> List[int]:
        """
        Update time for all parked cars (minute-based) and return customers that should leave.
        
        Args:
            current_timestep: Current timestep
        
        Returns:
            List of customer IDs that should leave
        """
        departing_customers = []
        
        for spot in self.spots.values():
            if spot.is_occupied and spot.arrival_timestep is not None:
                # Calculate if customer should leave based on duration
                minutes_parked = current_timestep - spot.arrival_timestep
                hours_parked = minutes_parked / 60.0
                
                if hours_parked >= spot.remaining_hours:
                    customer_id = spot.release_spot()
                    if customer_id is not None:
                        departing_customers.append(customer_id)
                        
        return departing_customers
        
    def get_zone_occupancy_vector(self) -> List[float]:
        """
        Get occupancy rates for all zones as a vector.
        
        Returns:
            List of occupancy rates [zone_A_rate, zone_B_rate, zone_C_rate]
        """
        occupancy_vector = []
        for zone in ['A', 'B', 'C']:
            occupied, total = self.get_zone_occupancy(zone)
            occupancy_rate = occupied / total if total > 0 else 0.0
            occupancy_vector.append(occupancy_rate)
        return occupancy_vector
        
    def get_queue_length(self) -> int:
        """
        Get current queue length.
        
        Returns:
            Number of customers in queue
        """
        return len(self.queue)
        
    def get_available_zones(self) -> List[str]:
        """
        Get list of zones with available spots.
        
        Returns:
            List of zone names with available spots
        """
        available_zones = []
        for zone in ['A', 'B', 'C']:
            if self.get_available_spots(zone):
                available_zones.append(zone)
        return available_zones
        
    def get_spot_info(self, spot_id: int) -> Optional[Dict]:
        """
        Get detailed information about a spot.
        
        Args:
            spot_id: Spot ID to get info for
            
        Returns:
            Dictionary with spot information or None if spot doesn't exist
        """
        if spot_id not in self.spots:
            return None
            
        spot = self.spots[spot_id]
        return {
            'spot_id': spot.spot_id,
            'zone': spot.zone,
            'is_occupied': spot.is_occupied,
            'customer_id': spot.customer_id,
            'remaining_hours': spot.remaining_hours,
            'arrival_timestep': spot.arrival_timestep
        }
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Get parking lot statistics.
        
        Returns:
            Dictionary with parking lot statistics
        """
        occupied, total = self.get_overall_occupancy()
        queue_length = len(self.queue)
        
        return {
            'occupancy_rate': occupied / total if total > 0 else 0.0,
            'queue_length': queue_length,
            'queue_utilization': queue_length / MAX_QUEUE_SIZE if MAX_QUEUE_SIZE > 0 else 0.0,
            'total_spots': self.total_spots,
            'occupied_spots': occupied,
            'available_spots': total - occupied
        } 