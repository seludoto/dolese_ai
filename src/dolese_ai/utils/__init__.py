"""
Utility modules for dolese_ai
"""

from .config import Config, load_config
from .logger import setup_logger, get_logger

config = Config()
logger = get_logger(__name__)

__all__ = ["Config", "load_config", "setup_logger", "get_logger", "config", "logger"]