"""
Classification models for dolese_ai
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..core import AIModel


class ClassificationModel(AIModel):
    """
    Classification model implementation with multiple algorithms.
    """
    
    def __init__(self, algorithm: str = "random_forest", config: Dict[str, Any] = None):
        """
        Initialize classification model.
        
        Args:
            algorithm: Algorithm to use ('random_forest', 'logistic_regression', 'svm')
            config: Configuration dictionary
        """
        super().__init__("classification", config)
        self.algorithm = algorithm
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model based on algorithm."""
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 100),
                random_state=self.config.get("random_state", 42)
            )
        elif self.algorithm == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.config.get("random_state", 42),
                max_iter=self.config.get("max_iter", 1000)
            )
        elif self.algorithm == "svm":
            self.model = SVC(
                random_state=self.config.get("random_state", 42),
                probability=True
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _fit_model(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit the classification model."""
        self.model.fit(X, y)
    
    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }