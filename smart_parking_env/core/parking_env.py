"""
Smart Parking Lot RL Environment with minute-based timesteps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random

# Core environment imports
from .config import (
    ZONE_CONFIG, TIMESTEPS_PER_EPISODE, MINUTES_PER_HOUR, REWARD_WEIGHTS,
    PRICE_LEVELS, DURATION_DISCOUNTS, PEAK_HOUR_MULTIPLIERS,
    ARRIVAL_RATES, ZONE_PREFERENCES, DURATION_TYPES, MAX_PARKING_DURATION,
    PRICE_CHANGE_CONFIG, VISUALIZATION_CONFIG, ACTION_SPACE_CONFIG, OBSERVATION_SPACE_CONFIG
)
from .customer import CustomerManager, Customer
from .parking_lot import ParkingLot
from .pricing import PricingManager


class SmartParkingEnv(gym.Env):
    """
    Smart Parking Lot RL Environment with minute-based timesteps.
    
    The agent makes single actions per timestep (minute) to process customers
    and manage pricing in a 50-spot parking lot over 24 hours (1440 timesteps).
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        # Initialize components
        self.customer_manager = CustomerManager()
        self.parking_lot = ParkingLot()
        self.pricing_manager = PricingManager()
        
        # Environment state
        self.current_timestep = 0
        self.render_mode = render_mode
        
        # Price change tracking
        self.price_changes_this_hour = 0
        self.last_price_change_timestep = -999
        
        # Action space: Single discrete action
        # 0=idle, 1=assign_A, 2=assign_B, 3=assign_C, 4=reject, 5=toggle_price_A, 6=toggle_price_B, 7=toggle_price_C
        self.action_space = spaces.Discrete(ACTION_SPACE_CONFIG['total_actions'])
        
        # Observation space: 13 dimensions
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SPACE_CONFIG['total_dimensions'],),
            dtype=np.float32
        )
        
        # Episode statistics
        self.episode_revenue = 0.0
        self.episode_rejections = 0
        self.episode_satisfaction = 0.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_timestep = 0
        self.price_changes_this_hour = 0
        self.last_price_change_timestep = -999
        
        # Reset components
        self.customer_manager = CustomerManager()
        self.parking_lot = ParkingLot()
        self.pricing_manager = PricingManager()
        
        # Reset episode statistics
        self.episode_revenue = 0.0
        self.episode_rejections = 0
        self.episode_satisfaction = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'hour': self.current_timestep // MINUTES_PER_HOUR,
            'minute': self.current_timestep % MINUTES_PER_HOUR,
            'total_revenue': self.episode_revenue,
            'rejections': self.episode_rejections
        }
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Single action (0-7)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        
        # Check if new hour started (reset price change counter)
        current_hour = self.current_timestep // MINUTES_PER_HOUR
        if self.current_timestep > 0 and self.current_timestep % MINUTES_PER_HOUR == 0:
            self.price_changes_this_hour = 0
        
        # Generate new customer if needed
        new_customer = self.customer_manager.generate_customer_if_needed(self.current_timestep)
        if new_customer:
            self.parking_lot.add_to_queue(new_customer)
        
        # Update wait times for queued customers
        queued_customers = list(self.parking_lot.queue)
        self.customer_manager.update_customer_wait_times(queued_customers)
        
        # Process action
        reward += self._process_action(action)
        
        # Update parking lot (handle departures)
        departing_customers = self.parking_lot.update_time_minute(self.current_timestep)
        for customer_id in departing_customers:
            customer = self.customer_manager.remove_customer(customer_id)
            if customer:
                self.episode_satisfaction += customer.satisfaction
        
        # Update timestep
        self.current_timestep += 1
        
        # Check if episode is done
        terminated = self.current_timestep >= TIMESTEPS_PER_EPISODE
        truncated = False
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _process_action(self, action: int) -> float:
        """
        Process the agent's action.
        
        Args:
            action: Action to process (0-7)
            
        Returns:
            Reward for this action
        """
        reward = 0.0
        
        if action == 0:  # Idle
            # No action taken
            pass
            
        elif action in [1, 2, 3, 4]:  # Customer processing actions
            # Get first customer from queue
            customer = self.parking_lot.get_next_queued_customer()
            
            if customer is not None:
                if action == 4:  # Reject customer
                    reward += self._reject_customer(customer)
                else:  # Assign to zone
                    zone_map = {1: 'A', 2: 'B', 3: 'C'}
                    target_zone = zone_map[action]
                    reward += self._assign_customer_to_zone(customer, target_zone)
            # If no customer in queue, action has no effect (no penalty)
            
        elif action in [5, 6, 7]:  # Price toggle actions
            zone_map = {5: 'A', 6: 'B', 7: 'C'}
            target_zone = zone_map[action]
            reward += self._toggle_zone_price(target_zone)
            
        return reward
        
    def _reject_customer(self, customer: Customer) -> float:
        """
        Reject a customer.
        
        Args:
            customer: Customer to reject
            
        Returns:
            Reward (penalty) for rejection
        """
        self.customer_manager.record_rejection()
        self.customer_manager.remove_customer(customer.customer_id)
        self.episode_rejections += 1
        
        # Check if spots are available (penalty only if rejecting when spots available)
        available_spots = self.parking_lot.get_total_available_spots()
        if available_spots > 0:
            return REWARD_WEIGHTS['rejection_penalty']
        else:
            return 0.0  # No penalty if no spots available
        
    def _assign_customer_to_zone(self, customer: Customer, target_zone: str) -> float:
        """
        Assign customer to a specific zone.
        
        Args:
            customer: Customer to assign
            target_zone: Zone to assign to ('A', 'B', 'C')
            
        Returns:
            Reward for assignment
        """
        available_spots = self.parking_lot.get_available_spots(target_zone)
        
        if not available_spots:
            # No spots available in target zone, put customer back in queue
            self.parking_lot.add_to_queue(customer)
            return 0.0  # No penalty for trying unavailable zone
        
        # Assign to first available spot
        spot_id = available_spots[0]
        
        # Calculate pricing with duration discount
        customer_discount = customer.get_duration_discount()
        total_price = self.pricing_manager.calculate_revenue(
            target_zone, customer.duration, customer_discount
        )
        
        # Assign customer to spot
        success = self.parking_lot.assign_customer_to_spot(
            customer, target_zone, spot_id, total_price, self.current_timestep
        )
        
        if success:
            # Calculate satisfaction
            base_price = self.pricing_manager.get_current_price(target_zone)
            satisfaction = customer.calculate_satisfaction(
                target_zone, total_price, base_price
            )
            
            # Record revenue
            self.pricing_manager.record_revenue(total_price)
            self.episode_revenue += total_price
            
            # Calculate reward
            reward = total_price * REWARD_WEIGHTS['revenue']
            reward += satisfaction * REWARD_WEIGHTS['satisfaction_bonus']
            
            return reward
        else:
            # Assignment failed, put back in queue
            self.parking_lot.add_to_queue(customer)
            return 0.0
        
    def _toggle_zone_price(self, zone: str) -> float:
        """
        Toggle price level for a zone.
        
        Args:
            zone: Zone to toggle price for
            
        Returns:
            Reward (penalty) for price change
        """
        # Check price change constraints
        current_hour = self.current_timestep // MINUTES_PER_HOUR
        minutes_since_last_change = self.current_timestep - self.last_price_change_timestep
        
        # Check if too many changes this hour
        if self.price_changes_this_hour >= PRICE_CHANGE_CONFIG['max_changes_per_hour']:
            return REWARD_WEIGHTS['price_fluctuation_penalty'] * 2  # Double penalty for exceeding limit
        
        # Check if too soon since last change
        if minutes_since_last_change < PRICE_CHANGE_CONFIG['min_minutes_between_changes']:
            return REWARD_WEIGHTS['price_fluctuation_penalty'] * 2  # Double penalty for frequent changes
        
        # Toggle price
        current_price_level = self.pricing_manager.get_price_level(zone)
        new_price_level = (current_price_level + 1) % 3  # Cycle through 0, 1, 2
        
        self.pricing_manager.set_price_level(zone, new_price_level)
        
        # Update tracking
        self.price_changes_this_hour += 1
        self.last_price_change_timestep = self.current_timestep
        
        # Return small penalty for price change
        return REWARD_WEIGHTS['price_fluctuation_penalty']
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation array with 12 dimensions
        """
        # Zone occupancy rates (3 dims) - already normalized 0-1
        occupancy_rates = self.parking_lot.get_zone_occupancy_vector()
        
        # Zone prices normalized (3 dims) - normalize by maximum possible price
        prices = self.pricing_manager.get_price_vector()
        # Maximum possible price is highest base price * highest multiplier
        max_possible_price = 8.0 * 1.3  # Zone A base * high price multiplier
        normalized_prices = [min(p / max_possible_price, 1.0) for p in prices]
        
        # Current hour normalized (1 dim)
        current_hour = self.current_timestep // MINUTES_PER_HOUR
        normalized_hour = min(current_hour / 24.0, 1.0)
        
        # Current minute normalized (1 dim)
        current_minute = self.current_timestep % MINUTES_PER_HOUR
        normalized_minute = current_minute / 60.0
        
        # Queue length normalized (1 dim)
        queue_length = self.parking_lot.get_queue_length()
        normalized_queue = min(queue_length / 10.0, 1.0)  # Cap at 1.0
        
        # First customer wait time normalized (1 dim)
        first_customer = None
        if self.parking_lot.queue:
            first_customer = self.parking_lot.queue[0]
        first_wait_time = first_customer.wait_time if first_customer else 0
        normalized_first_wait = min(first_wait_time / 60.0, 1.0)  # Normalize by 1 hour, cap at 1.0
        
        # Total queue wait time normalized (1 dim)
        total_wait_time = sum(c.wait_time for c in self.parking_lot.queue)
        normalized_total_wait = min(total_wait_time / 300.0, 1.0)  # Normalize by 5 hours, cap at 1.0
        
        # Price changes this hour normalized (1 dim)
        max_changes = PRICE_CHANGE_CONFIG['max_changes_per_hour']
        normalized_price_changes = min(self.price_changes_this_hour / max_changes, 1.0) if max_changes > 0 else 0.0
        
        # Time since last price change normalized (1 dim)
        minutes_since_last_change = self.current_timestep - self.last_price_change_timestep
        normalized_time_since_change = min(minutes_since_last_change / 60.0, 1.0)  # Normalize by 1 hour
        
        # Combine into observation vector (12 dimensions)
        observation = np.array([
            *occupancy_rates,           # 3 dims
            *normalized_prices,         # 3 dims
            normalized_hour,            # 1 dim
            normalized_minute,          # 1 dim
            normalized_queue,           # 1 dim
            normalized_first_wait,      # 1 dim
            normalized_total_wait,      # 1 dim
            normalized_price_changes,   # 1 dim
            normalized_time_since_change # 1 dim
        ], dtype=np.float32)
        
        # Ensure all values are between 0 and 1
        observation = np.clip(observation, 0.0, 1.0)
        
        return observation
        
    def _get_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dictionary with environment information
        """
        current_hour = self.current_timestep // MINUTES_PER_HOUR
        current_minute = self.current_timestep % MINUTES_PER_HOUR
        occupied, total = self.parking_lot.get_overall_occupancy()
        queue_length = self.parking_lot.get_queue_length()
        
        # Get customer statistics and flatten them into the main info dict
        info = self.customer_manager.get_statistics()
        
        info.update({
            'hour': current_hour,
            'minute': current_minute,
            'timestep': self.current_timestep,
            'total_revenue': self.episode_revenue,
            'rejections': self.episode_rejections,
            'occupancy_rate': occupied / total if total > 0 else 0.0,
            'queue_length': queue_length,
            'zone_occupancy': self.parking_lot.get_zone_occupancy_vector(),
            'zone_prices': self.pricing_manager.get_price_vector(),
            'price_changes_this_hour': self.price_changes_this_hour,
        })
        
        return info
        
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            Rendered frame or None
        """
        if self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            return np.zeros((800, 1200, 3), dtype=np.uint8)
        else:
            return None
            
    def close(self) -> None:
        """Close the environment."""
        pass
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        customer_stats = self.customer_manager.get_statistics()
        
        return {
            **customer_stats,
            'episode_revenue': self.episode_revenue,
            'episode_rejections': self.episode_rejections,
            'episode_satisfaction': self.episode_satisfaction,
            'total_timesteps': self.current_timestep
        } 