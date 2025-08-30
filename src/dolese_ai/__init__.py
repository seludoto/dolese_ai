"""
Dolese AI - Artificial Intelligence Development Framework

A modern Python framework for AI/ML development with focus on:
- Machine Learning model development
- Natural Language Processing
- Computer Vision
- Data Processing and Analysis
- Model Training and Evaluation

Example usage:
    >>> from dolese_ai import AIModel
    >>> model = AIModel()
    >>> model.train(data)
"""

__version__ = "0.1.0"
__author__ = "Dolese AI Team"
__email__ = "team@dolese.ai"

# Core imports
from .core import AIModel, DataProcessor
from .utils import config, logger
from .models import *

__all__ = [
    "AIModel",
    "DataProcessor", 
    "config",
    "logger",
    "__version__"
]