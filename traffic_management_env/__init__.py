"""
Traffic Management Environment - A Gymnasium-compatible environment for traffic light control.
"""

from gymnasium.envs.registration import register

register(
    id='TrafficManagement-v0',
    entry_point='traffic_management_env.environment:TrafficManagementEnv',
    max_episode_steps=1000,
    kwargs={
        'grid_size': (5, 5),
        'num_intersections': 9,
        'max_vehicles': 50,
        'spawn_rate': 0.3,
    }
)

__version__ = "1.0.0"