"""
World Builder Game Environment

A simple resource management world builder game using Gymnasium environment
with Pygame rendering.

The agent manages resources (Food, Wood, Stone, Population) and builds
structures to grow their population while maintaining resource balance.
"""

from .world_builder_env import WorldBuilderEnv

__version__ = "1.0.0"
__all__ = ["WorldBuilderEnv"] 