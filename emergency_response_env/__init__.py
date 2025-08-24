"""
Emergency Response Environment Package

A comprehensive emergency response coordination environment for reinforcement learning
that simulates realistic disaster scenarios with multiple emergency services,
complex resource allocation, and time-critical decision making.

Author: Hasnain Fareed
License: MIT (2025)
"""

from gymnasium.envs.registration import register
from .emergency_env import EmergencyResponseEnv, EmergencyType, UnitType

__version__ = "1.0.0"
__author__ = "Hasnain Fareed"
__email__ = ""
__license__ = "MIT"

# Register the environment with Gymnasium
register(
    id="EmergencyResponse-v0",
    entry_point="emergency_response_env.emergency_env:EmergencyResponseEnv",
    max_episode_steps=1440,  # 24 hours in minutes
    kwargs={
        "render_mode": None
    }
)

register(
    id="EmergencyResponse-v1", 
    entry_point="emergency_response_env.emergency_env:EmergencyResponseEnv",
    max_episode_steps=1440,
    kwargs={
        "render_mode": "human"
    }
)

# Make key classes available at package level
__all__ = [
    "EmergencyResponseEnv",
    "EmergencyType", 
    "UnitType",
]

# Package metadata
DESCRIPTION = """
Emergency Response Environment for Reinforcement Learning

Features:
- 30x30 metropolitan area simulation
- 40+ emergency response units (Fire, Police, Ambulance, Search & Rescue, Hazmat, Command)
- 8 emergency types with realistic response requirements
- Complex state space with 276 observations
- 50 discrete actions for comprehensive emergency management
- Pygame visualization with real-time incident tracking
- Cascade effects and emergency escalation
- Weather and traffic conditions
- Multi-agency coordination challenges
- Resource fatigue and readiness modeling
- Hospital capacity management
- Communication network simulation
- Mass casualty and disaster scenarios
- Performance metrics and after-action reporting
"""

def get_env_info():
    """Get environment information"""
    return {
        "name": "Emergency Response Environment",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": DESCRIPTION.strip(),
        "registered_envs": [
            "EmergencyResponse-v0",  # No rendering
            "EmergencyResponse-v1"   # With human rendering
        ]
    }

def create_env(render_mode=None):
    """Create an EmergencyResponseEnv instance"""
    return EmergencyResponseEnv(render_mode=render_mode)

# Environment creation shortcuts
def make_env_headless():
    """Create environment without rendering"""
    return create_env(render_mode=None)

def make_env_visual():
    """Create environment with pygame visualization"""
    return create_env(render_mode="human")

def make_env_rgb():
    """Create environment that returns RGB arrays"""
    return create_env(render_mode="rgb_array")

# Package information
print(f"Emergency Response Environment v{__version__} loaded successfully!")
print(f"Available environments: {', '.join(['EmergencyResponse-v0', 'EmergencyResponse-v1'])}")
print("Use gymnasium.make('EmergencyResponse-v0') to create an environment instance.")




