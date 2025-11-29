"""
Detectors package for MK8DX table reader.
Contains base classes and implementations for player (names & scores) and table detection (table & players).
"""

from .base_player_detector import BasePlayerDetector
from . import yoloUseModelTable
from . import onnx_use_model_players

__all__ = [
    'BasePlayerDetector',
    'yoloUseModelTable',
    'onnx_use_model_players'
]
