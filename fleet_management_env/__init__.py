"""
Fleet Management Environment Package

A comprehensive multi-agent reinforcement learning environment for urban delivery logistics.
Features dynamic traffic, multi-vehicle coordination, and realistic delivery constraints.
"""

from gymnasium.envs.registration import register
from .fleet_env import FleetManagementEnv

__version__ = "1.0.0"
__author__ = "Fleet Management Environment"

# Register the environment with Gymnasium
register(
    id='FleetManagement-v0',
    entry_point='fleet_management_env:FleetManagementEnv',
    max_episode_steps=800,
    reward_threshold=2000.0,
)

__all__ = ['FleetManagementEnv']