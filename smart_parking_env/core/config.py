"""
Configuration constants for the Smart Parking Lot RL Environment.
"""

# Zone configurations
ZONE_CONFIG = {
    'A': {'spots': 15, 'base_price': 8.0, 'name': 'Premium'},
    'B': {'spots': 20, 'base_price': 5.0, 'name': 'Standard'}, 
    'C': {'spots': 15, 'base_price': 3.0, 'name': 'Economy'}
}

# Time parameters - Updated for minute-based system
HOURS_PER_EPISODE = 24
MINUTES_PER_HOUR = 60
TIMESTEPS_PER_EPISODE = HOURS_PER_EPISODE * MINUTES_PER_HOUR  # 1440 timesteps
TIMESTEP_MINUTES = 1  # 1 minute per timestep

# Customer arrival rates (customers per hour) - converted to per minute rates
ARRIVAL_RATES = {
    6: 8,    # 6 AM - Morning rush
    7: 12,
    8: 15,
    9: 10,
    10: 8,   # Mid-morning
    11: 8,
    12: 12,  # Lunch
    13: 12,
    14: 8,
    15: 8,
    16: 10,
    17: 15,  # Evening rush
    18: 18,
    19: 15,
    20: 12,
    21: 8,
    22: 6,
    23: 4,
    0: 3,    # Late night
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 4
}

# Convert hourly arrival rates to per-minute probabilities
ARRIVAL_PROBABILITIES = {hour: rate / MINUTES_PER_HOUR for hour, rate in ARRIVAL_RATES.items()}

# Duration types and probabilities (in hours)
DURATION_TYPES = {
    'short': {'min_hours': 1, 'max_hours': 2, 'probability': 0.4},
    'medium': {'min_hours': 3, 'max_hours': 5, 'probability': 0.35},
    'long': {'min_hours': 6, 'max_hours': 12, 'probability': 0.25}
}

# Duration-based pricing discounts
DURATION_DISCOUNTS = {
    'short': 0.0,    # No discount for short stays (1-2h)
    'medium': 0.15,  # 15% discount for medium stays (3-5h)
    'long': 0.25     # 25% discount for long stays (6+h)
}

# Peak hour multipliers for dynamic pricing
PEAK_HOUR_MULTIPLIERS = {
    'A': {'peak': 1.3, 'off_peak': 0.8},
    'B': {'peak': 1.2, 'off_peak': 0.9},
    'C': {'peak': 1.1, 'off_peak': 0.95}
}

# Maximum parking duration (24 hours)
MAX_PARKING_DURATION = 24

# Zone preference probabilities
ZONE_PREFERENCES = {
    'A': 0.3,  # 30% prefer premium
    'B': 0.35, # 35% prefer standard
    'C': 0.15, # 15% prefer economy
    'flexible': 0.2  # 20% are flexible
}

# Pricing levels (multipliers of base price)
PRICE_LEVELS = {
    0: 0.7,  # Low price (30% discount)
    1: 1.0,  # Medium price (base price)
    2: 1.3   # High price (30% premium)
}

# Queue configuration
MAX_QUEUE_SIZE = 10  # Increased for minute-based system

# Price fluctuation control
PRICE_CHANGE_CONFIG = {
    'max_changes_per_hour': 2,
    'min_minutes_between_changes': 15,
    'fluctuation_penalty': 0.5
}

# Updated reward weights - balanced system
REWARD_WEIGHTS = {
    'revenue': 1.0,
    'rejection_penalty': -2.0,
    'price_fluctuation_penalty': -0.5,
    'satisfaction_bonus': 0.8
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'window_width': 1200,
    'window_height': 800,
    'grid_size': 40,
    'colors': {
        'zone_a': (255, 215, 0),  # Gold
        'zone_b': (0, 255, 0),    # Green
        'zone_c': (0, 0, 255),    # Blue
        'empty': (255, 255, 255), # White
        'occupied': (255, 0, 0),  # Red
        'queued': (255, 255, 0),  # Yellow
        'background': (50, 50, 50), # Dark gray
        'text': (255, 255, 255)   # White
    },
    'font_size': 16,
    'info_panel_width': 300
}

# Updated action space configuration
ACTION_SPACE_CONFIG = {
    'idle': 0,
    'assign_zone_a': 1,
    'assign_zone_b': 2,
    'assign_zone_c': 3,
    'reject_customer': 4,
    'toggle_price_a': 5,
    'toggle_price_b': 6,
    'toggle_price_c': 7,
    'total_actions': 8
}

# Updated observation space configuration
OBSERVATION_SPACE_CONFIG = {
    'zone_occupancy': 3,              # Occupancy for each zone
    'zone_prices': 3,                 # Current prices for each zone
    'hour': 1,                        # Current hour (0-23)
    'minute': 1,                      # Current minute (0-59)
    'queue_length': 1,                # Number of customers in queue
    'first_customer_wait_time': 1,    # Wait time of first customer
    'total_queue_wait_time': 1,       # Total wait time in queue
    'price_change_count': 1,          # Price changes this hour
    'time_since_last_price_change': 1, # Minutes since last price change
    'total_dimensions': 13
} 