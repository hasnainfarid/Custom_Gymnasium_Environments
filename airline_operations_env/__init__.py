"""
Airline Operations Environment Package

A comprehensive Gymnasium environment for airline operations management
with hub-and-spoke network simulation.

Copyright (c) 2025 Hasnain Fareed
Licensed under the MIT License
"""

from gymnasium.envs.registration import register
from .airline_env import AirlineOperationsEnv

# Register the environment
register(
    id='AirlineOperations-v0',
    entry_point='airline_operations_env.airline_env:AirlineOperationsEnv',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

# Export main classes
__all__ = [
    'AirlineOperationsEnv',
]

__version__ = '1.0.0'
__author__ = 'Hasnain Fareed'
__license__ = 'MIT'




