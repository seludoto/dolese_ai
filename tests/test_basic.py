"""
Basic test for dolese_ai package
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from dolese_ai import AIModel, DataProcessor
from dolese_ai.models import ClassificationModel
from dolese_ai.utils import Config


def test_basic_import():
    """Test basic imports work correctly."""
    from dolese_ai import AIModel, DataProcessor
    assert AIModel is not None
    assert DataProcessor is not None


def test_config():
    """Test configuration management."""
    config = Config()
    
    # Test default values
    assert config.get("model.type") == "base"
    assert config.get("model.random_state") == 42
    
    # Test setting values
    config.set("model.type", "test")
    assert config.get("model.type") == "test"


def test_data_processor():
    """Test data processing functionality."""
    processor = DataProcessor()
    
    # Create test data
    data = pd.DataFrame({
        'feature_1': [1, 2, np.nan, 4, 5],
        'feature_2': [10, np.nan, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Test data cleaning
    cleaned = processor.clean_data(data)
    assert cleaned.isna().sum().sum() == 0  # No missing values


def test_classification_model():
    """Test classification model."""
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    
    # Create model
    model = ClassificationModel(algorithm="random_forest")
    
    # Test training
    model.train(X, y)
    assert model.is_trained
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    
    # Test evaluation
    metrics = model.evaluate(X[:50], y[:50])
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])