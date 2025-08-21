"""Smart Manufacturing Environment Package

A comprehensive manufacturing simulation environment for reinforcement learning
that models a factory floor with production stations, quality control, and
machine maintenance requirements.
"""

from gym.envs.registration import register
from .manufacturing_env import SmartManufacturingEnv

# Register the environment with OpenAI Gym
register(
    id='SmartManufacturing-v0',
    entry_point='smart_manufacturing_env:SmartManufacturingEnv',
    max_episode_steps=1500,
    reward_threshold=10000.0,
)

__version__ = '1.0.0'
__all__ = ['SmartManufacturingEnv']