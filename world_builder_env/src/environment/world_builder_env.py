import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union

from .game_logic import GameLogic
from .renderer import Renderer


class WorldBuilderEnv(gym.Env):
    """
    A simple resource management world builder game.
    
    The agent manages resources (Food, Wood, Stone, Population) and builds
    structures to grow their population while maintaining resource balance.
    
    Actions:
        0: Pass (do nothing)
        1: Build Farm (costs 5 wood, produces 2 food per step)
        2: Build Lumberyard (costs 3 stone, produces 3 wood per step)
        3: Build Quarry (costs 5 wood, produces 2 stone per step)
        4: Build House (costs 10 wood + 5 stone, +5 population capacity)
    
    Observations:
        grid: 10x10 grid showing building types (0=empty, 1=farm, 2=lumberyard, 3=quarry, 4=house)
        resources: [food, wood, stone, population] current resource amounts
        population_capacity: maximum population that can be supported
        win_steps: steps since reaching 20 population (for win condition)
    
    Rewards:
        +100: Win (reach 20 population and survive 50 steps)
        -100: Lose (population drops to 0)
        -50: Other termination
        +small rewards: for successful building, population growth, resource balance
        -small penalties: for failed builds, resource waste, population loss
    """
    
    def __init__(self, grid_size: int = 10, render_mode: Optional[str] = None, flatten_obs: bool = False):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.flatten_obs = flatten_obs  # For RLlib DQN compatibility
        
        # Game constants
        self.MAX_POPULATION = 20
        self.WIN_STEPS = 50
        self.BUILDING_TYPES = {
            0: "empty",
            1: "farm",      # 5 wood, produces 2 food
            2: "lumberyard", # 3 stone, produces 3 wood  
            3: "quarry",     # 5 wood, produces 2 stone
            4: "house"       # 10 wood + 5 stone, +5 population capacity
        }
        
        # Action space: 5 actions (4 building types + pass)
        self.action_space = spaces.Discrete(5)
        
        # Action mapping for clarity (following Gymnasium example)
        self._action_to_building = {
            0: "pass",
            1: "farm",
            2: "lumberyard", 
            3: "quarry",
            4: "house"
        }
        
        # Observation space: grid + resources + population info
        if self.flatten_obs:
            # Flattened observation for RLlib DQN compatibility
            # Grid (flattened): 100 values + Resources: 4 + Population capacity: 1 + Win steps: 1 = 106 total
            obs_size = grid_size * grid_size + 4 + 1 + 1
            self.observation_space = spaces.Box(
                low=0.0, 
                high=1000.0, 
                shape=(obs_size,), 
                dtype=np.float32
            )
        else:
            # Dict observation for better interpretability
            self.observation_space = spaces.Dict({
                'grid': spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.int8),
                'resources': spaces.Box(low=0.0, high=1000.0, shape=(4,), dtype=np.float32),  # Consistent float32
                'population_capacity': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),  # Consistent float32
                'win_steps': spaces.Box(low=0, high=self.WIN_STEPS, shape=(1,), dtype=np.int32)
            })
        
        # Initialize game logic and renderer
        self.game_logic = GameLogic(grid_size, self.BUILDING_TYPES)
        self.renderer = None
        if render_mode == "human":
            self.renderer = Renderer(grid_size)
        
        # Game state
        self.steps = 0
        self.win_steps = 0
        self.reached_win_population = False
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
            
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.win_steps = 0
        self.reached_win_population = False
        
        # Initialize game logic
        self.game_logic.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.
        
        Args:
            action: The action to take (0-4 for building types + pass)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action space is {self.action_space}")
        
        self.steps += 1
        
        # Execute action
        reward = self.game_logic.execute_action(action)
        
        # Update win condition tracking
        if self.game_logic.population >= self.MAX_POPULATION and not self.reached_win_population:
            self.reached_win_population = True
        
        if self.reached_win_population:
            self.win_steps += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False  # We don't use truncation in this environment
        
        # Calculate final reward
        if terminated:
            if self.game_logic.population <= 0:
                reward = -100  # Lose condition
            elif self.win_steps >= self.WIN_STEPS:
                reward = 100   # Win condition
            else:
                reward = -50   # Other termination
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.renderer is not None:
            return self.renderer.render(
                self.game_logic.grid,
                self.game_logic.resources,
                self.game_logic.population,
                self.game_logic.population_capacity,
                self.steps,
                self.win_steps
            )
        return None
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
    
    def _get_observation(self):
        """Convert internal state to observation format.
        
        Returns:
            Union[dict, np.ndarray]: Observation (dict for normal use, flattened array for RLlib DQN)
        """
        if self.flatten_obs:
            # Flatten the observation for RLlib DQN compatibility
            grid_flat = self.game_logic.grid.flatten().astype(np.float32)  # Convert to float32
            resources = np.array([
                self.game_logic.resources['food'],
                self.game_logic.resources['wood'],
                self.game_logic.resources['stone'],
                self.game_logic.population
            ], dtype=np.float32)
            population_capacity = np.array([self.game_logic.population_capacity], dtype=np.float32)
            win_steps = np.array([self.win_steps], dtype=np.float32)  # Convert to float32 for consistency
            return np.concatenate([grid_flat, resources, population_capacity, win_steps]).astype(np.float32)
        else:
            return {
                'grid': self.game_logic.grid.copy(),
                'resources': np.array([
                    self.game_logic.resources['food'],
                    self.game_logic.resources['wood'],
                    self.game_logic.resources['stone'],
                    self.game_logic.population
                ], dtype=np.float32),
                'population_capacity': np.array([self.game_logic.population_capacity], dtype=np.float32),
                'win_steps': np.array([self.win_steps], dtype=np.int32)
            }
    
    def _get_info(self) -> Dict[str, Any]:
        """Compute auxiliary information for debugging.
        
        Returns:
            dict: Info with game state details
        """
        return {
            'steps': self.steps,
            'win_steps': self.win_steps,
            'reached_win_population': self.reached_win_population,
            'resources': self.game_logic.resources.copy(),
            'population': self.game_logic.population,
            'population_capacity': self.game_logic.population_capacity,
            'building_counts': self.game_logic.building_counts.copy()
        }
    
    def _check_termination(self) -> bool:
        """Check if the episode should terminate.
        
        Returns:
            bool: True if episode should end
        """
        # Lose condition: population drops to 0
        if self.game_logic.population <= 0:
            return True
        
        # Win condition: reach 20 population and survive for 50 steps
        if self.reached_win_population and self.win_steps >= self.WIN_STEPS:
            return True
        
        return False 