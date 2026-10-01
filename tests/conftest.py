"""Shared fixtures: a small synthetic frame that follows the BRFSS schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic(n_rows: int = 5_000, positive_rate: float = 0.14, seed: int = 0) -> pd.DataFrame:
    """Random rows inside every column's valid range, in raw column order."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in config.RAW_COLUMNS:
        lo, hi = config.VALUE_RANGES[col]
        data[col] = rng.integers(lo, hi + 1, size=n_rows)
    data[config.TARGET] = (rng.random(n_rows) < positive_rate).astype(int)
    return pd.DataFrame(data)[config.RAW_COLUMNS].astype(np.int64)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()
