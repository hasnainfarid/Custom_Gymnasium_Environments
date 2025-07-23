"""
Pricing management for the Smart Parking Lot RL Environment.
"""

import random
from typing import Dict, Optional, List

# Paperspace-compatible imports
try:
    from .config import ZONE_CONFIG, PRICE_LEVELS
except ImportError:
    from config import ZONE_CONFIG, PRICE_LEVELS


class PricingManager:
    """
    Manages pricing logic and calculations for the parking lot.
    """
    
    def __init__(self):
        """Initialize the pricing manager."""
        self.current_prices: Dict[str, float] = {}
        self.price_levels: Dict[str, int] = {'A': 1, 'B': 1, 'C': 1}  # Default to medium
        self.total_revenue = 0.0
        
        # Initialize with base prices
        self._update_prices()
        
    def set_price_level(self, zone: str, level: int) -> bool:
        """
        Set price level for a specific zone.
        
        Args:
            zone: Zone to set price for
            level: Price level (0=low, 1=medium, 2=high)
            
        Returns:
            True if successful, False otherwise
        """
        if zone not in ZONE_CONFIG or level not in PRICE_LEVELS:
            return False
            
        self.price_levels[zone] = level
        self._update_prices()
        return True
        
    def get_price_level(self, zone: str) -> int:
        """
        Get current price level for a zone.
        
        Args:
            zone: Zone to get price level for
            
        Returns:
            Price level (0=low, 1=medium, 2=high)
        """
        return self.price_levels.get(zone, 1)  # Default to medium
        
    def _update_prices(self) -> None:
        """Update current prices based on price levels."""
        for zone, config in ZONE_CONFIG.items():
            level = self.price_levels[zone]
            multiplier = PRICE_LEVELS[level]
            self.current_prices[zone] = config['base_price'] * multiplier
                
    def get_current_price(self, zone: str) -> float:
        """
        Get current price for a zone.
        
        Args:
            zone: Zone to get price for
            
        Returns:
            Current price for the zone
        """
        return self.current_prices.get(zone, 0.0)
        
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get current prices for all zones.
        
        Returns:
            Dictionary mapping zone to current price
        """
        return self.current_prices.copy()
        
    def get_price_vector(self) -> List[float]:
        """
        Get current prices as a vector.
        
        Returns:
            List of prices [zone_A_price, zone_B_price, zone_C_price]
        """
        return [self.current_prices.get(zone, 0.0) for zone in ['A', 'B', 'C']]
        
    def calculate_revenue(self, zone: str, duration: int, customer_discount: float = 1.0) -> float:
        """
        Calculate revenue for a parking session.
        
        Args:
            zone: Zone where customer parks
            duration: Duration of stay in hours
            customer_discount: Customer-specific discount multiplier
            
        Returns:
            Total revenue for the session
        """
        base_price = self.get_current_price(zone)
        
        # Apply duration-based pricing (full hour billing)
        duration_multiplier = self._get_duration_multiplier(duration)
        
        # Calculate hourly rate with duration discount
        hourly_rate = base_price * duration_multiplier * customer_discount
        
        # Calculate total revenue (full hours billing)
        total_revenue = hourly_rate * duration
        
        return total_revenue
        
    def _get_duration_multiplier(self, duration: int) -> float:
        """
        Get duration-based pricing multiplier.
        
        Args:
            duration: Duration of stay in hours
            
        Returns:
            Pricing multiplier
        """
        if duration >= 6:  # Long duration - 25% discount
            return 0.75
        elif duration >= 3:  # Medium duration - 15% discount
            return 0.85
        else:  # Short duration - no discount
            return 1.0
        
    def get_price_level_description(self, level: int) -> str:
        """
        Get description for a price level.
        
        Args:
            level: Price level (0, 1, or 2)
            
        Returns:
            Description of the price level
        """
        descriptions = {
            0: "Low",
            1: "Medium", 
            2: "High"
        }
        return descriptions.get(level, "Unknown")
        
    def get_zone_price_info(self, zone: str) -> Dict[str, float]:
        """
        Get detailed price information for a zone.
        
        Args:
            zone: Zone to get info for
            
        Returns:
            Dictionary with price information
        """
        if zone not in ZONE_CONFIG:
            return {}
            
        config = ZONE_CONFIG[zone]
        level = self.price_levels[zone]
        current_price = self.current_prices[zone]
        
        return {
            'zone': zone,
            'base_price': config['base_price'],
            'price_level': level,
            'price_level_name': self.get_price_level_description(level),
            'current_price': current_price,
            'multiplier': PRICE_LEVELS[level]
        }
        
    def get_all_zone_price_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed price information for all zones.
        
        Returns:
            Dictionary mapping zone to price information
        """
        return {zone: self.get_zone_price_info(zone) for zone in ['A', 'B', 'C']}
        
    def record_revenue(self, amount: float) -> None:
        """
        Record revenue from a transaction.
        
        Args:
            amount: Revenue amount to record
        """
        self.total_revenue += amount
        
    def get_total_revenue(self) -> float:
        """
        Get total revenue recorded.
        
        Returns:
            Total revenue
        """
        return self.total_revenue
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Get pricing statistics.
        
        Returns:
            Dictionary with pricing statistics
        """
        avg_price = sum(self.current_prices.values()) / len(self.current_prices) if self.current_prices else 0.0
        min_price = min(self.current_prices.values()) if self.current_prices else 0.0
        max_price = max(self.current_prices.values()) if self.current_prices else 0.0
        
        return {
            'total_revenue': self.total_revenue,
            'average_price': avg_price,
            'min_price': min_price,
            'max_price': max_price,
            'price_variance': max_price - min_price
        } 