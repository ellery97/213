"""Prediction helpers."""
from __future__ import annotations

import numpy as np

from .model import TimeSeriesModel


def predict(model: TimeSeriesModel, X: np.ndarray) -> np.ndarray:
    """Generate predictions using a trained model."""
    return model.predict(X)
