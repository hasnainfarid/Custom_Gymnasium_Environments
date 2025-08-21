"""Space Station Life Support Management Environment Package"""

from gymnasium.envs.registration import register
from .station_env import SpaceStationEnv

# Register the environment
register(
    id='SpaceStation-v0',
    entry_point='space_station_env.station_env:SpaceStationEnv',
    max_episode_steps=8640,  # 30 days * 24 hours * 60 minutes / 5 minutes per step
    reward_threshold=100000.0,
)

__version__ = "1.0.0"
__all__ = ['SpaceStationEnv']