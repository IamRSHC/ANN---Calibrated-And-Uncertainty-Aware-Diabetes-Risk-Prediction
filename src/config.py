"""Project-wide constants: random seed, paths, dataset schema and split settings.

Every module imports from here so that the seed, file locations and feature
groups are defined exactly once.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed Python's and NumPy's global generators.

    Components with their own ``random_state``/``seed`` argument (sklearn,
    XGBoost, splits, CV, the scratch model) still receive ``SEED`` explicitly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"

RAW_FILE: Path = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"

# File names inside PROCESSED_DIR. Only src/features.py writes TEST_FILE and
# only src/evaluate.py reads it (enforced by tests/test_test_set_access.py).
TRAIN_FILE: str = "train.csv"
TEST_FILE: str = "test.csv"
TEST_INDEX_FILE: str = "test_indices.csv"
SPLIT_MANIFEST_FILE: str = "split_manifest.json"
TEST_LOCK_FILE: str = "TEST_LOCK"

# ---------------------------------------------------------------------------
# Dataset schema (CDC BRFSS 2015, binary diabetes indicators)
# ---------------------------------------------------------------------------
TARGET: str = "Diabetes_binary"
EXPECTED_ROWS: int = 253_680

BINARY_FEATURES: list[str] = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
CONTINUOUS_FEATURES: list[str] = ["BMI", "MentHlth", "PhysHlth"]
ORDINAL_FEATURES: list[str] = ["GenHlth", "Age", "Education", "Income"]
FEATURES: list[str] = BINARY_FEATURES + CONTINUOUS_FEATURES + ORDINAL_FEATURES

# Column order of the raw CSV (target first, then features as published).
RAW_COLUMNS: list[str] = [
    TARGET,
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

# Inclusive valid ranges. Binary columns (and the target) must be exactly 0/1.
VALUE_RANGES: dict[str, tuple[int, int]] = {
    TARGET: (0, 1),
    **{col: (0, 1) for col in BINARY_FEATURES},
    "BMI": (12, 98),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

# ---------------------------------------------------------------------------
# Split settings
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
MAX_CLASS_RATIO_GAP: float = 0.005  # Phase 0 gate: +/-0.5 percentage points
