"""Model definitions for time series forecasting."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression


class TimeSeriesModel:
    """A simple wrapper around scikit-learn's LinearRegression."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the regression model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts."""
        return self.model.predict(X)
