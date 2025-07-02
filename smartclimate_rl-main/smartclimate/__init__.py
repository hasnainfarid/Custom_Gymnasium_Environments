import gymnasium
from .env import SmartClimateEnv

import smartclimate  # This ensures the env is registered!

gymnasium.register(
    id='SmartClimateEnv-v0',
    entry_point='smartclimate.env:SmartClimateEnv',
    max_episode_steps=1440
) 

# RLlib registration (for direct RLlib usage)
try:
    from ray.tune.registry import register_env
    def env_creator(env_config=None):
        from environment import OfficeEnergyEnv
        return OfficeEnergyEnv()
    register_env("SmartClimateEnv-v0", env_creator)
except ImportError:
    pass  # Ray not installed, skip RLlib registration 