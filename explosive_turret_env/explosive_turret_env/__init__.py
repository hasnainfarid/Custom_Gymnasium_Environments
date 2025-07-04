"""
Explosive Turret Environment
A realistic Box2D + Pygame environment where a turret shoots explosive shells at destructible targets.
"""

from .turret_env import TurretEnv
from gymnasium.envs.registration import register

# Register the environment with Gymnasium
register(
    id="ExplosiveTurret-v1",
    entry_point="explosive_turret_env:TurretEnv",
    max_episode_steps=600,
    reward_threshold=None,
    nondeterministic=False,
)

__version__ = "1.0.0"
__author__ = "Hasnain Fareed"
__email__ = "Hasnainfarid7@yahoo.com"

__all__ = ["TurretEnv"] 