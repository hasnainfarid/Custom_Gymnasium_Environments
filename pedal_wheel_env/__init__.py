"""
Pedal Wheel Environment - A Gymnasium environment for pedal-controlled unicycle simulation.
"""

from .environment import PedalWheelEnv
from .physics import PedalWheelPhysics
from .config import *

__version__ = "1.0.0"
__author__ = "Custom Environment Creator"

# Main environment class
__all__ = ["PedalWheelEnv", "PedalWheelPhysics"] 