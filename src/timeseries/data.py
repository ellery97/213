"""Data loading utilities."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load time series data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the time series.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.
    """
    return pd.read_csv(filepath)
