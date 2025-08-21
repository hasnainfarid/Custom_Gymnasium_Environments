"""
Hospital Management Environment Package
A realistic hospital operations simulation for reinforcement learning.
"""

from .hospital_env import HospitalManagementEnv
from gymnasium.envs.registration import register

__version__ = "1.0.0"
__author__ = "Hospital Management AI Team"

# Register the environment with Gymnasium
register(
    id='HospitalManagement-v0',
    entry_point='hospital_management_env:HospitalManagementEnv',
    max_episode_steps=1440,  # 24 hours in minutes
    reward_threshold=10000.0,
)

__all__ = ['HospitalManagementEnv']