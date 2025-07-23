"""
Smart Parking Lot RL Environment Package
"""

# Paperspace-compatible imports
try:
    from .core.parking_env import SmartParkingEnv
except ImportError:
    from core.parking_env import SmartParkingEnv

__version__ = "1.0.0"
__all__ = ['SmartParkingEnv'] 