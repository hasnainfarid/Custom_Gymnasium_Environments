"""
Configuration parameters for the Traffic Management Environment.
"""

# Grid and intersection settings
DEFAULT_GRID_SIZE = (5, 5)  # 5x5 grid of intersections
DEFAULT_NUM_INTERSECTIONS = 9  # Total intersections to control
INTERSECTION_SPACING = 100  # Distance between intersections (meters)

# Vehicle settings
MAX_VEHICLES = 50  # Maximum number of vehicles in the environment
DEFAULT_SPAWN_RATE = 0.3  # Probability of spawning a vehicle per timestep
VEHICLE_SPEED = 10  # Base vehicle speed (m/s)
VEHICLE_LENGTH = 5  # Vehicle length (meters)

# Traffic light settings
LIGHT_PHASES = ['NS_GREEN', 'NS_YELLOW', 'EW_GREEN', 'EW_YELLOW']  # Traffic light phases
MIN_PHASE_DURATION = 5  # Minimum duration for each phase (timesteps)
MAX_PHASE_DURATION = 30  # Maximum duration for each phase (timesteps)
YELLOW_DURATION = 3  # Fixed duration for yellow lights (timesteps)

# Simulation settings
MAX_TIMESTEPS = 1000  # Maximum episode length
TIMESTEP_DURATION = 1.0  # Duration of each timestep (seconds)

# Reward settings
REWARD_VEHICLE_PASSED = 1.0  # Reward for each vehicle that passes through intersection
PENALTY_WAITING_TIME = -0.1  # Penalty per timestep per waiting vehicle
PENALTY_QUEUE_LENGTH = -0.05  # Penalty per vehicle in queue
REWARD_SMOOTH_FLOW = 0.5  # Reward for maintaining smooth traffic flow
PENALTY_EMERGENCY_STOP = -5.0  # Penalty for emergency stops

# Observation settings
MAX_QUEUE_LENGTH = 20  # Maximum queue length to observe
OBSERVATION_RADIUS = 2  # Number of intersections around each intersection to observe

# Rendering settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
INTERSECTION_SIZE = 60
VEHICLE_SIZE = 8
COLORS = {
    'BACKGROUND': (50, 50, 50),
    'ROAD': (100, 100, 100),
    'INTERSECTION': (80, 80, 80),
    'VEHICLE': (255, 255, 0),
    'EMERGENCY_VEHICLE': (255, 0, 0),
    'GREEN_LIGHT': (0, 255, 0),
    'YELLOW_LIGHT': (255, 255, 0),
    'RED_LIGHT': (255, 0, 0),
    'TEXT': (255, 255, 255),
}