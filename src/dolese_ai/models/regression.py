"""
Regression models for dolese_ai
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..core import AIModel


class RegressionModel(AIModel):
    """
    Regression model implementation with multiple algorithms.
    """
    
    def __init__(self, algorithm: str = "random_forest", config: Dict[str, Any] = None):
        """
        Initialize regression model.
        
        Args:
            algorithm: Algorithm to use ('random_forest', 'linear_regression', 'svr')
            config: Configuration dictionary
        """
        super().__init__("regression", config)
        self.algorithm = algorithm
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model based on algorithm."""
        if self.algorithm == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.get("n_estimators", 100),
                random_state=self.config.get("random_state", 42)
            )
        elif self.algorithm == "linear_regression":
            self.model = LinearRegression()
        elif self.algorithm == "svr":
            self.model = SVR()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _fit_model(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit the regression model."""
        self.model.fit(X, y)
    
    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }