"""
Summarization module for AI Notes Assistant.

This module provides AI-powered content enhancement and summarization
capabilities using various local and remote models.
"""

from .llama_local import MixtralProcessor
from .custom_model import CustomModelProcessor
from .utils import *

__all__ = [
    'MixtralProcessor',
    'CustomModelProcessor',
]
