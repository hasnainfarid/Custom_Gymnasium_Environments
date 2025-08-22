import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any


class SnakeEnvClassic(gym.Env):
    """
    A Snake game environment for reinforcement learning.
    
    The agent controls a snake that moves around a grid, eating food to grow longer.
    The game ends if the snake hits the wall or itself.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 20):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid state + snake direction + food direction
        # Grid: 0=empty, 1=snake, 2=food
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(grid_size, grid_size), dtype=np.int8
        )
        
        # Initialize pygame for rendering
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * 20, grid_size * 20))
            pygame.display.set_caption("Snake RL Environment")
            self.clock = pygame.time.Clock()
        
        # Game state
        self.snake = None
        self.food = None
        self.direction = None
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize snake at center
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = 1  # Start moving right
        self.score = 0
        self.steps = 0
        
        # Place food
        self._place_food()
        
        observation = self._get_observation()
        info = {"score": self.score, "snake_length": len(self.snake)}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Update direction (prevent 180-degree turns)
        if abs(action - self.direction) != 2:
            self.direction = action
        
        # Move snake
        head = self.snake[0]
        if self.direction == 0:  # Up
            new_head = (head[0] - 1, head[1])
        elif self.direction == 1:  # Right
            new_head = (head[0], head[1] + 1)
        elif self.direction == 2:  # Down
            new_head = (head[0] + 1, head[1])
        else:  # Left
            new_head = (head[0], head[1] - 1)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return self._get_observation(), -10.0, True, False, {"score": self.score}
        
        # Check collision with self
        if new_head in self.snake:
            return self._get_observation(), -10.0, True, False, {"score": self.score}
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
        
        self.steps += 1
        
        # Check if episode should end
        done = False
        if self.steps >= self.max_steps:
            done = True
        
        observation = self._get_observation()
        info = {"score": self.score, "snake_length": len(self.snake)}
        
        return observation, reward, done, False, info
    
    def _place_food(self):
        """Place food at a random empty location."""
        while True:
            self.food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if self.food not in self.snake:
                break
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation as a grid."""
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Mark snake positions
        for segment in self.snake:
            obs[segment[0], segment[1]] = 1
        
        # Mark food position
        if self.food:
            obs[self.food[0], self.food[1]] = 2
        
        return obs
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render using pygame."""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * 20, i * 20, 20, 20)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw snake
        for segment in self.snake:
            rect = pygame.Rect(segment[1] * 20, segment[0] * 20, 20, 20)
            pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Green snake
        
        # Draw food
        if self.food:
            rect = pygame.Rect(self.food[1] * 20, self.food[0] * 20, 20, 20)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red food
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self) -> np.ndarray:
        """Return RGB array for rendering."""
        # Create a simple RGB array representation
        obs = self._get_observation()
        rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Empty cells: black
        rgb[obs == 0] = [0, 0, 0]
        # Snake: green
        rgb[obs == 1] = [0, 255, 0]
        # Food: red
        rgb[obs == 2] = [255, 0, 0]
        
        return rgb
    
    def close(self):
        """Close the environment."""
        if self.render_mode == "human":
            pygame.quit() 