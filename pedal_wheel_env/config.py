"""
Configuration constants for the Pedal Wheel Environment.
"""

import numpy as np

# Physics constants
GRAVITY = 9.81  # m/s²
WHEEL_RADIUS = 0.5  # meters
WHEEL_MASS = 10.0  # kg
FRAME_MASS = 5.0  # kg
TOTAL_MASS = WHEEL_MASS + FRAME_MASS

# Pedal constants
PEDAL_LENGTH = 0.3  # meters
MAX_PEDAL_FORCE = 100.0  # Newtons
PEDAL_FRICTION = 0.1  # Friction coefficient

# Balance and movement
MAX_TILT_ANGLE = np.pi / 4  # 45 degrees in radians
MAX_FORWARD_VELOCITY = 10.0  # m/s
MAX_ANGULAR_VELOCITY = 5.0  # rad/s

# Environment bounds
WORLD_WIDTH = 100.0  # meters
WORLD_HEIGHT = 20.0  # meters
GROUND_Y = 15.0  # meters from top

# Episode settings
MAX_STEPS = 1000
TIME_STEP = 0.05  # seconds per step

# Reward parameters
REWARD_PER_STEP = 1.0
FALL_PENALTY = -10.0
VELOCITY_BONUS_FACTOR = 0.1
ENERGY_PENALTY_FACTOR = 0.01

# Visualization
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIXELS_PER_METER = 5.0  # Scale factor for visualization

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19) 