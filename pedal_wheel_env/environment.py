"""
Main Gymnasium environment for the Pedal Wheel Environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
try:
    from .physics import PedalWheelPhysics
    from .config import *
except ImportError:
    from physics import PedalWheelPhysics
    from config import *


class PedalWheelEnv(gym.Env):
    """
    A Gymnasium environment for a pedal-controlled unicycle.
    
    The agent controls two pedals to move and balance a single-wheeled vehicle
    on a flat 2D road. The goal is to stay upright and move forward efficiently.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    
    def __init__(self, render_mode=None):
        """
        Initialize the Pedal Wheel Environment.
        
        Args:
            render_mode (str, optional): Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.physics = PedalWheelPhysics()
        self.step_count = 0
        
        # Action space: [left_pedal_force, right_pedal_force]
        # Each pedal force is in the range [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [position, velocity, tilt_angle, angular_velocity, wheel_angular_velocity]
        # All values are normalized to [-1.0, 1.0] range
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Initialize renderer if needed
        self.renderer = None
        if self.render_mode == "human":
            try:
                try:
                    from .pygame_visualizer import PygameVisualizer
                except ImportError:
                    from pygame_visualizer import PygameVisualizer
                self.renderer = PygameVisualizer()
            except ImportError:
                print("Warning: Pygame not available. Human rendering disabled.")
                self.render_mode = None
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset physics
        self.physics.reset()
        self.step_count = 0
        
        # Get initial observation
        observation = self.physics.get_observation()
        
        info = {
            "step_count": self.step_count,
            "position": self.physics.x,
            "velocity": self.physics.vx,
            "tilt_angle": self.physics.theta,
            "energy_used": self.physics.total_energy_used
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Array of shape (2,) containing [left_pedal_force, right_pedal_force]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Extract pedal forces
        left_pedal_force, right_pedal_force = action
        
        # Advance physics simulation
        self.physics.step(left_pedal_force, right_pedal_force)
        self.step_count += 1
        
        # Get observation and reward
        observation = self.physics.get_observation()
        reward = self.physics.get_reward()
        
        # Check termination conditions
        terminated = self.physics.is_fallen()
        truncated = self.step_count >= MAX_STEPS
        
        # Create info dictionary
        info = {
            "step_count": self.step_count,
            "position": self.physics.x,
            "velocity": self.physics.vx,
            "tilt_angle": self.physics.theta,
            "energy_used": self.physics.total_energy_used,
            "fallen": terminated,
            "max_steps_reached": truncated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human" and self.renderer is not None:
            return self.renderer.render(self.physics.get_state())
        elif self.render_mode == "rgb_array":
            # Return a simple RGB array representation
            # This is a placeholder - in practice, you'd want to draw the scene
            return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        else:
            return None
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
    
    def get_state(self):
        """
        Get the current physics state.
        
        Returns:
            dict: Current state information
        """
        return self.physics.get_state()
    
    def set_state(self, state):
        """
        Set the physics state (for debugging/testing).
        
        Args:
            state (dict): State dictionary
        """
        # This is a simplified implementation
        # In practice, you'd want to validate the state and set all physics variables
        for key, value in state.items():
            if hasattr(self.physics, key):
                setattr(self.physics, key, value) 