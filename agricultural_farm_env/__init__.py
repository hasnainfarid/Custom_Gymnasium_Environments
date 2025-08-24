"""
Agricultural Farm Management Environment for Gymnasium.

A comprehensive farming simulation environment with crop management,
weather systems, market dynamics, and sustainability tracking.
"""

from gymnasium.envs.registration import register

from .farm_env import AgriculturalFarmEnv

# Register the environment
register(
    id='AgriculturalFarm-v0',
    entry_point='agricultural_farm_env.farm_env:AgriculturalFarmEnv',
    max_episode_steps=365,  # One year of farming
    reward_threshold=10000.0,
)

__all__ = ['AgriculturalFarmEnv']

__version__ = '1.0.0'

