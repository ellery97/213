"""Training utilities."""
from __future__ import annotations

import numpy as np

from .model import TimeSeriesModel


def train_model(X: np.ndarray, y: np.ndarray) -> TimeSeriesModel:
    """Train a TimeSeriesModel with the provided data."""
    model = TimeSeriesModel()
    model.fit(X, y)
    return model
