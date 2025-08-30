"""
Configuration management for dolese_ai
"""

import os
import json
from typing import Any, Dict, Optional
from pathlib import Path

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class Config:
    """
    Configuration management class for dolese_ai.
    
    Supports loading configuration from JSON files and environment variables.
    YAML support is optional (requires PyYAML).
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (JSON or YAML if available)
        """
        self._config = {}
        self._load_defaults()
        
        if config_file:
            self.load_file(config_file)
        
        self._load_env_vars()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            "model": {
                "type": "base",
                "random_state": 42,
                "verbose": True
            },
            "data": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "early_stopping": True,
                "patience": 10
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.type' or 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.type')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "DOLESE_AI_LOG_LEVEL": ("logging", "level"),
            "DOLESE_AI_MODEL_TYPE": ("model", "type"),
            "DOLESE_AI_BATCH_SIZE": ("training", "batch_size"),
            "DOLESE_AI_LEARNING_RATE": ("training", "learning_rate"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = value


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_file)