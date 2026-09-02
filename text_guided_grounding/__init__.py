"""
Text-Guided Grounding Specialist Model Module (SIH26 PS-2 / SatQuery)

This module provides vision-language text-guided localization and visual grounding
for satellite imagery based on natural language referring expressions.
"""

from .inference import ground
from .model import TextGuidedGroundingModel
from .dataset import RSVGDataset

__all__ = ["ground", "TextGuidedGroundingModel", "RSVGDataset"]
