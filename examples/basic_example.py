#!/usr/bin/env python3
"""
Simple example demonstrating dolese_ai usage
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Import dolese_ai components
from dolese_ai import AIModel, DataProcessor
from dolese_ai.models import ClassificationModel, RegressionModel
from dolese_ai.utils import setup_logger

def main():
    """Run example demonstrations."""
    # Set up logging
    logger = setup_logger("dolese_ai_example", level="INFO")
    logger.info("Starting dolese_ai example")
    
    # Example 1: Basic Classification
    logger.info("=== Classification Example ===")
    
    # Generate sample data
    X_class, y_class = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    
    # Create DataFrame for better handling
    feature_names = [f"feature_{i}" for i in range(X_class.shape[1])]
    df_class = pd.DataFrame(X_class, columns=feature_names)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(df_class, y_class)
    
    # Create and train classification model
    clf = ClassificationModel(algorithm="random_forest")
    clf.train(X_train, y_train)
    
    # Make predictions and evaluate
    predictions = clf.predict(X_test)
    metrics = clf.evaluate(X_test, y_test)
    
    logger.info(f"Classification Accuracy: {metrics['accuracy']:.4f}")
    
    # Example 2: Basic Regression
    logger.info("=== Regression Example ===")
    
    # Generate sample regression data
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X_reg.shape[1])]
    df_reg = pd.DataFrame(X_reg, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(df_reg, y_reg)
    
    # Create and train regression model
    reg = RegressionModel(algorithm="random_forest")
    reg.train(X_train, y_train)
    
    # Make predictions and evaluate
    predictions = reg.predict(X_test)
    metrics = reg.evaluate(X_test, y_test)
    
    logger.info(f"Regression R²: {metrics['r2']:.4f}")
    logger.info(f"Regression RMSE: {metrics['rmse']:.4f}")
    
    # Example 3: Data Processing
    logger.info("=== Data Processing Example ===")
    
    # Create sample data with missing values
    sample_data = pd.DataFrame({
        'feature_1': [1, 2, np.nan, 4, 5],
        'feature_2': [10, np.nan, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    logger.info(f"Original data shape: {sample_data.shape}")
    logger.info("Original data:")
    logger.info(f"\n{sample_data}")
    
    # Clean data
    cleaned_data = processor.clean_data(sample_data)
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    logger.info("Cleaned data:")
    logger.info(f"\n{cleaned_data}")
    
    logger.info("dolese_ai example completed successfully!")

if __name__ == "__main__":
    main()