"""
Crop data and dynamics for the agricultural farm environment.
All data is self-contained within this module, no external agricultural databases used.
"""

from .crop_data import CROP_DATA, CropType, CropStage

__all__ = ['CROP_DATA', 'CropType', 'CropStage']

