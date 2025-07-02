"""
Configuration constants for the Bus System Environment.
"""

# Environment parameters
NUM_STOPS = 4
NUM_BUSES = 4
BUS_CAPACITY = 20
MAX_PASSENGERS = 150
MAX_TIMESTEPS = 500
MAX_DWELL_TIME = 10

# Movement parameters
TRAVEL_TIME_BETWEEN_STOPS = 5  # timesteps to travel between stops

# Reward weights
REWARD_PER_DELIVERY = 5.0
PENALTY_PER_WAITING_PASSENGER = -1.0
PENALTY_PER_ONBOARD_PASSENGER = -0.5

# Visualization parameters
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
STOP_RADIUS = 30
BUS_SIZE = 20
FONT_SIZE = 16

# Colors (RGB tuples)
COLORS = {
    'background': (255, 255, 255),      # White
    'route': (100, 100, 100),           # Gray
    'stop': (0, 0, 255),                # Blue
    'bus': (255, 0, 0),                 # Red
    'passenger': (0, 255, 0),           # Green
    'text': (0, 0, 0),                  # Black
    'waiting_passengers': (255, 165, 0), # Orange
} 