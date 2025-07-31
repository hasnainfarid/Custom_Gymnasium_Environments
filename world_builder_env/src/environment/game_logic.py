import numpy as np
from typing import Dict, Tuple, Optional


class GameLogic:
    """
    Handles the core game logic for resource management and building mechanics.
    """
    
    def __init__(self, grid_size: int, building_types: Dict[int, str]):
        self.grid_size = grid_size
        self.building_types = building_types
        
        # Building costs and production
        self.building_costs = {
            'farm': {'wood': 5},
            'lumberyard': {'stone': 3},
            'quarry': {'wood': 5},
            'house': {'wood': 10, 'stone': 5}
        }
        
        self.building_production = {
            'farm': {'food': 2},
            'lumberyard': {'wood': 3},
            'quarry': {'stone': 2},
            'house': {'population_capacity': 5}
        }
        
        # Initialize game state
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Initial resources
        self.resources = {
            'food': 25,  # Balanced initial food
            'wood': 20,  # More wood for building farms
            'stone': 10
        }
        
        # Population
        self.population = 3  # Start with smaller population
        self.population_capacity = 10
        
        # Track building counts for production
        self.building_counts = {
            'farm': 0,
            'lumberyard': 0,
            'quarry': 0,
            'house': 0
        }
    
    def execute_action(self, action: int) -> float:
        """
        Execute an action and return a smart reward.
        """
        prev_population = self.population
        prev_food = self.resources['food']
        prev_capacity = self.population_capacity
        prev_building_counts = self.building_counts.copy()
        reward = 0
        
        # --- Building action rewards ---
        if action == 0:  # Pass
            reward += 0
        else:
            building_type = list(self.building_types.values())[action]
            if building_type != "empty":
                success = self._try_build(building_type)
                if success:
                    if building_type == 'farm':
                        reward += 3
                    elif building_type == 'lumberyard':
                        reward += 2
                    elif building_type == 'quarry':
                        reward += 2
                    elif building_type == 'house':
                        reward += 4
                    # Bonus for reaching capacity
                    if building_type == 'house' and prev_population >= prev_capacity - 1:
                        reward += 10
                else:
                    reward -= 3  # Higher penalty for failed building
        
        # --- Process production and consumption ---
        self._process_production()
        self._process_consumption()
        self._process_population_growth()
        
        # --- Survival and growth rewards ---
        # Reward for population growth
        if self.population > prev_population:
            reward += 5
        # Penalty for population loss
        if self.population < prev_population:
            reward -= 50
        # Reward for food surplus
        if self.resources['food'] > self.population * 2:
            reward += 1
        # Penalty for food deficit
        if self.resources['food'] < self.population:
            reward -= 2
        # Penalty for close to starvation
        if self.resources['food'] < max(2, self.population):
            reward -= 5
        # Reward for resource balance
        if abs(self.resources['wood'] - self.resources['stone']) < 5:
            reward += 1
        # Penalty for wasteful action (building farm with lots of food, etc.)
        if action == 1 and self.resources['food'] > self.population * 3:
            reward -= 1
        
        return reward
    
    def _try_build(self, building_type: str) -> bool:
        """Try to build a structure at a random empty location."""
        # Check if we have enough resources
        if not self._can_afford_building(building_type):
            return False
        
        # Find empty locations
        empty_positions = np.where(self.grid == 0)
        if len(empty_positions[0]) == 0:
            return False  # No empty space
        
        # Choose random empty position
        idx = np.random.randint(len(empty_positions[0]))
        row, col = empty_positions[0][idx], empty_positions[1][idx]
        
        # Deduct resources
        self._spend_resources(building_type)
        
        # Place building
        building_id = list(self.building_types.keys())[list(self.building_types.values()).index(building_type)]
        self.grid[row, col] = building_id
        
        # Update building counts
        self.building_counts[building_type] += 1
        
        # Add population capacity for houses
        if building_type == 'house':
            self.population_capacity += 5
        
        return True
    
    def _can_afford_building(self, building_type: str) -> bool:
        """Check if we have enough resources to build."""
        costs = self.building_costs[building_type]
        for resource, amount in costs.items():
            if self.resources[resource] < amount:
                return False
        return True
    
    def _spend_resources(self, building_type: str):
        """Spend resources for building."""
        costs = self.building_costs[building_type]
        for resource, amount in costs.items():
            self.resources[resource] -= amount
    
    def _process_production(self):
        """Process production from all buildings."""
        for building_type, count in self.building_counts.items():
            if count > 0:
                production = self.building_production[building_type]
                for resource, amount in production.items():
                    if resource == 'population_capacity':
                        # Population capacity is added when building is built, not every step
                        pass
                    else:
                        self.resources[resource] += amount * count
    
    def _process_consumption(self):
        """Process food consumption by population."""
        food_needed = self.population
        if self.resources['food'] < food_needed:
            # Not enough food - population dies
            self.population = 0
        else:
            self.resources['food'] -= food_needed
    
    def _process_population_growth(self):
        """Process population growth based on excess food."""
        if self.population <= 0:
            return  # Population already died
        
        excess_food = self.resources['food']
        if excess_food > 2 and self.population < self.population_capacity:  # Need more excess food
            # Population grows by 1 if there's excess food and capacity
            self.population += 1
            self.resources['food'] -= 1  # Consume the food for growth
    
    def get_state_summary(self) -> Dict:
        """Get a summary of the current game state."""
        return {
            'resources': self.resources.copy(),
            'population': self.population,
            'population_capacity': self.population_capacity,
            'building_counts': self.building_counts.copy(),
            'grid': self.grid.copy()
        } 