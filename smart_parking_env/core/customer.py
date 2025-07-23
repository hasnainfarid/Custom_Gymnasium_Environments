"""
Customer class and logic for the Smart Parking Lot RL Environment.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple

# Paperspace-compatible imports
try:
    from .config import DURATION_TYPES, ZONE_PREFERENCES, ARRIVAL_PROBABILITIES, MAX_PARKING_DURATION
except ImportError:
    from config import DURATION_TYPES, ZONE_PREFERENCES, ARRIVAL_PROBABILITIES, MAX_PARKING_DURATION


class Customer:
    """
    Represents a customer in the parking lot system.
    """
    
    def __init__(self, customer_id: int, arrival_time: int, zone_preference: str, 
                 duration: int, satisfaction: float = 1.0):
        """
        Initialize a customer.
        
        Args:
            customer_id: Unique identifier for the customer
            arrival_time: Timestep when customer arrived
            zone_preference: Preferred zone ('A', 'B', 'C', or 'flexible')
            duration: Duration of stay in hours
            satisfaction: Customer satisfaction level (0.0 to 1.0)
        """
        self.customer_id = customer_id
        self.arrival_time = arrival_time
        self.zone_preference = zone_preference
        self.duration = min(duration, MAX_PARKING_DURATION)  # Cap at 24 hours
        self.satisfaction = satisfaction
        self.assigned_zone = None
        self.assigned_spot = None
        self.price_paid = 0.0
        self.is_queued = False
        self.queue_position = -1
        self.wait_time = 0  # In minutes
        
    def assign_spot(self, zone: str, spot_id: int, price: float) -> None:
        """
        Assign a parking spot to the customer.
        
        Args:
            zone: Zone where spot is assigned
            spot_id: ID of the assigned spot
            price: Price paid for the spot
        """
        self.assigned_zone = zone
        self.assigned_spot = spot_id
        self.price_paid = price
        self.is_queued = False
        self.queue_position = -1
        
    def add_to_queue(self, position: int) -> None:
        """
        Add customer to queue.
        
        Args:
            position: Position in queue
        """
        self.is_queued = True
        self.queue_position = position
        
    def remove_from_queue(self) -> None:
        """Remove customer from queue."""
        self.is_queued = False
        self.queue_position = -1
        
    def update_wait_time(self, minutes: int = 1) -> None:
        """
        Update customer wait time.
        
        Args:
            minutes: Minutes to add to wait time
        """
        self.wait_time += minutes
        
    def calculate_satisfaction(self, assigned_zone: str, price: float, 
                             base_price: float) -> float:
        """
        Calculate customer satisfaction based on assignment and pricing.
        
        Args:
            assigned_zone: Zone where customer was assigned
            price: Price paid
            base_price: Base price for the zone
            
        Returns:
            Satisfaction score (0.0 to 1.0)
        """
        satisfaction = 1.0
        
        # Zone preference satisfaction
        if self.zone_preference == 'flexible':
            satisfaction *= 0.9  # Slightly lower for flexible customers
        elif assigned_zone == self.zone_preference:
            satisfaction *= 1.0  # Perfect match
        else:
            satisfaction *= 0.6  # Preference mismatch
            
        # Price satisfaction
        price_ratio = price / base_price
        if price_ratio <= 0.8:  # Good discount
            satisfaction *= 1.1
        elif price_ratio <= 1.0:  # Fair price
            satisfaction *= 1.0
        elif price_ratio <= 1.2:  # Slightly expensive
            satisfaction *= 0.8
        else:  # Very expensive
            satisfaction *= 0.5
            
        # Wait time penalty (decrease satisfaction based on wait time)
        if self.wait_time > 0:
            wait_penalty = min(self.wait_time / 60.0, 0.3)  # Max 30% penalty
            satisfaction *= (1.0 - wait_penalty)
            
        self.satisfaction = max(0.0, min(1.0, satisfaction))
        return self.satisfaction
        
    def get_duration_discount(self) -> float:
        """
        Get duration-based discount multiplier.
        
        Returns:
            Discount multiplier (0.8 = 20% discount, 1.0 = no discount)
        """
        if self.duration >= 6:  # Long duration
            return 0.75  # 25% discount
        elif self.duration >= 3:  # Medium duration
            return 0.85  # 15% discount
        else:  # Short duration
            return 1.0  # No discount
            
    def __str__(self) -> str:
        """String representation of customer."""
        status = "Queued" if self.is_queued else f"Zone {self.assigned_zone}" if self.assigned_zone else "Unassigned"
        return f"Customer {self.customer_id}: {status}, Duration: {self.duration}h, Satisfaction: {self.satisfaction:.2f}"


def generate_customer(customer_id: int, current_hour: int) -> Customer:
    """
    Generate a new customer with time-based preferences.
    
    Args:
        customer_id: Unique identifier for the customer
        current_hour: Current hour of the day
        
    Returns:
        Customer object with random preferences
    """
    # Determine zone preference with time-based adjustments
    adjusted_preferences = get_adjusted_zone_preferences(current_hour)
    
    rand = random.random()
    cumulative_prob = 0
    zone_preference = 'flexible'  # Default
    
    for zone, prob in adjusted_preferences.items():
        cumulative_prob += prob
        if rand <= cumulative_prob:
            zone_preference = zone
            break
    
    # Generate duration with time-based patterns
    duration_type = get_time_based_duration_type(current_hour)
    config = DURATION_TYPES[duration_type]
    duration = random.randint(config['min_hours'], config['max_hours'])
    
    # Add some randomness to duration
    if random.random() < 0.3:  # 30% chance of extended duration
        duration += random.randint(1, 3)
    
    return Customer(
        customer_id=customer_id,
        arrival_time=current_hour,
        zone_preference=zone_preference,
        duration=duration
    )


def get_adjusted_zone_preferences(hour: int) -> Dict[str, float]:
    """
    Get zone preferences adjusted for time of day.
    
    Args:
        hour: Current hour (0-23)
        
    Returns:
        Adjusted zone preferences
    """
    base_preferences = ZONE_PREFERENCES.copy()
    
    if 6 <= hour <= 9:  # Morning rush
        # More premium preferences during rush hour
        base_preferences['A'] *= 1.3
        base_preferences['B'] *= 1.1
        base_preferences['C'] *= 0.8
        base_preferences['flexible'] *= 0.9
    elif 17 <= hour <= 19:  # Evening rush
        # More standard preferences during evening
        base_preferences['A'] *= 0.9
        base_preferences['B'] *= 1.2
        base_preferences['C'] *= 1.1
        base_preferences['flexible'] *= 1.0
    elif 22 <= hour or hour <= 5:  # Late night
        # More flexible during late hours
        base_preferences['A'] *= 0.7
        base_preferences['B'] *= 0.8
        base_preferences['C'] *= 1.0
        base_preferences['flexible'] *= 1.5
    
    # Normalize probabilities
    total = sum(base_preferences.values())
    return {zone: prob / total for zone, prob in base_preferences.items()}


def get_time_based_duration_type(hour: int) -> str:
    """
    Get duration type based on time of day.
    
    Args:
        hour: Current hour (0-23)
        
    Returns:
        Duration type ('short', 'medium', 'long')
    """
    if 6 <= hour <= 9:  # Morning rush - more short stays
        return 'short' if random.random() < 0.6 else 'medium'
    elif 12 <= hour <= 14:  # Lunch time - balanced
        return random.choice(['short', 'medium', 'long'])
    elif 17 <= hour <= 19:  # Evening rush - more medium stays
        return 'medium' if random.random() < 0.6 else 'long'
    else:  # Off-peak - more long stays
        return 'long' if random.random() < 0.5 else 'medium'


def should_customer_arrive(current_hour: int) -> bool:
    """
    Determine if a customer should arrive at current timestep (minute).
    
    Args:
        current_hour: Current hour (0-23)
        
    Returns:
        True if customer should arrive
    """
    arrival_probability = ARRIVAL_PROBABILITIES.get(current_hour, 0.05)
    return random.random() < arrival_probability


class CustomerManager:
    """
    Manages customer generation and tracking for minute-based system.
    """
    
    def __init__(self):
        """Initialize customer manager."""
        self.next_customer_id = 0
        self.customers: Dict[int, Customer] = {}
        self.total_customers = 0
        self.rejected_customers = 0
        self.satisfied_customers = 0
        self.total_wait_time = 0
        
    def generate_customer_if_needed(self, current_timestep: int) -> Optional[Customer]:
        """
        Generate a customer if arrival conditions are met.
        
        Args:
            current_timestep: Current timestep (0-1439)
            
        Returns:
            New customer or None
        """
        current_hour = current_timestep // 60  # Convert timestep to hour
        
        if should_customer_arrive(current_hour):
            customer = generate_customer(self.next_customer_id, current_timestep)
            self.customers[customer.customer_id] = customer
            self.next_customer_id += 1
            self.total_customers += 1
            return customer
        
        return None
        
    def update_customer_wait_times(self, queued_customers: List[Customer]) -> None:
        """
        Update wait times for queued customers.
        
        Args:
            queued_customers: List of customers in queue
        """
        for customer in queued_customers:
            customer.update_wait_time(1)  # Add 1 minute
            self.total_wait_time += 1
        
    def get_customer(self, customer_id: int) -> Optional[Customer]:
        """
        Get customer by ID.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Customer object or None if not found
        """
        return self.customers.get(customer_id)
        
    def remove_customer(self, customer_id: int) -> Optional[Customer]:
        """
        Remove customer from tracking.
        
        Args:
            customer_id: Customer ID to remove
            
        Returns:
            Removed customer or None if not found
        """
        customer = self.customers.pop(customer_id, None)
        if customer and customer.satisfaction > 0.7:
            self.satisfied_customers += 1
        return customer
        
    def record_rejection(self) -> None:
        """Record a customer rejection."""
        self.rejected_customers += 1
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Get customer statistics.
        
        Returns:
            Dictionary with customer statistics
        """
        if self.total_customers == 0:
            return {
                'total_customers': 0,
                'rejection_rate': 0.0,
                'satisfaction_rate': 0.0,
                'avg_wait_time': 0.0
            }
            
        return {
            'total_customers': self.total_customers,
            'rejection_rate': self.rejected_customers / self.total_customers,
            'satisfaction_rate': self.satisfied_customers / self.total_customers,
            'avg_wait_time': self.total_wait_time / max(1, self.total_customers)
        } 