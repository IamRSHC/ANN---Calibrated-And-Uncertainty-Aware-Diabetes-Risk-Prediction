"""Data loading, schema validation, duplicate analysis and the train/test split.

This is the single home for all preprocessing and feature engineering. The
notebooks and the Streamlit app import from here; nothing re-implements these
steps elsewhere.

The test split is written by :func:`save_split` and is deliberately never read
by this module. The only reader is ``src/evaluate.py::evaluate_on_test``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


class SchemaError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


# ---------------------------------------------------------------------------
# Loading and schema validation
# ---------------------------------------------------------------------------
@dataclass
class SchemaReport:
    """Outcome of :func:`validate_schema`."""

    n_rows: int
    n_columns: int
    columns_match: bool
    column_order_matches: bool
    missing_columns: list[str]
    unexpected_columns: list[str]
    raw_dtypes: dict[str, str]
    n_missing_values: dict[str, int]
    non_integer_values: dict[str, int]
    out_of_range_values: dict[str, int]
    observed_ranges: dict[str, tuple[float, float]]
    row_count_matches_expected: bool
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no blocking schema error was found."""
        return not self.errors

    def summary_table(self) -> pd.DataFrame:
        """Per-column table of dtype, observed range, expected range and issues."""
        rows = []
        for col in config.RAW_COLUMNS:
            if col not in self.raw_dtypes:
                continue
            lo, hi = config.VALUE_RANGES[col]
            obs_lo, obs_hi = self.observed_ranges[col]
            rows.append(
                {
                    "column": col,
                    "group": _column_group(col),
                    "raw_dtype": self.raw_dtypes[col],
                    "observed_min": obs_lo,
                    "observed_max": obs_hi,
                    "expected_min": lo,
                    "expected_max": hi,
                    "missing": self.n_missing_values[col],
                    "non_integer": self.non_integer_values[col],
                    "out_of_range": self.out_of_range_values[col],
                }
            )
        return pd.DataFrame(rows).set_index("column")


def _column_group(col: str) -> str:
    if col == config.TARGET:
        return "target"
    if col in config.BINARY_FEATURES:
        return "binary"
    if col in config.CONTINUOUS_FEATURES:
        return "continuous"
    return "ordinal"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (used to pin the raw CSV version)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_csv(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw BRFSS CSV exactly as published, without any conversion.

    Raises:
        FileNotFoundError: if the CSV has not been downloaded into ``data/raw/``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download "
            f"'{config.RAW_FILE.name}' (Kaggle: 'Diabetes Health Indicators "
            f"Dataset', CDC BRFSS 2015) and place it in {config.RAW_DIR}."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> SchemaReport:
    """Check columns, dtypes, missing values, integer-ness and value ranges.

    Every column in this dataset is integer-coded (BMI included), so any
    non-integer value is reported as an error, as is any value outside
    ``config.VALUE_RANGES``.
    """
    errors: list[str] = []
    expected = config.RAW_COLUMNS
    missing_cols = [c for c in expected if c not in df.columns]
    unexpected_cols = [c for c in df.columns if c not in expected]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
    if unexpected_cols:
        errors.append(f"unexpected columns: {unexpected_cols}")

    present = [c for c in expected if c in df.columns]
    raw_dtypes = {c: str(df[c].dtype) for c in present}
    n_missing: dict[str, int] = {}
    non_integer: dict[str, int] = {}
    out_of_range: dict[str, int] = {}
    observed: dict[str, tuple[float, float]] = {}

    for col in present:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            errors.append(f"{col}: non-numeric dtype {series.dtype}")
            series = pd.to_numeric(series, errors="coerce")
        n_missing[col] = int(series.isna().sum())
        values = series.dropna().to_numpy(dtype=np.float64)
        non_integer[col] = int(np.sum(values != np.round(values)))
        lo, hi = config.VALUE_RANGES[col]
        out_of_range[col] = int(np.sum((values < lo) | (values > hi)))
        observed[col] = (
            (float(values.min()), float(values.max())) if values.size else (np.nan, np.nan)
        )
        if n_missing[col]:
            errors.append(f"{col}: {n_missing[col]} missing values")
        if non_integer[col]:
            errors.append(f"{col}: {non_integer[col]} non-integer values")
        if out_of_range[col]:
            errors.append(f"{col}: {out_of_range[col]} values outside [{lo}, {hi}]")

    return SchemaReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        columns_match=not missing_cols and not unexpected_cols,
        column_order_matches=list(df.columns) == expected,
        missing_columns=missing_cols,
        unexpected_columns=unexpected_cols,
        raw_dtypes=raw_dtypes,
        n_missing_values=n_missing,
        non_integer_values=non_integer,
        out_of_range_values=out_of_range,
        observed_ranges=observed,
        row_count_matches_expected=len(df) == config.EXPECTED_ROWS,
        errors=errors,
    )


def cast_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in canonical column order with every column as int64.

    Only call this after :func:`validate_schema` reported no errors, which
    guarantees every value is a finite integer within range.
    """
    return df[config.RAW_COLUMNS].astype(np.int64)


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read, validate and cast the raw dataset.

    Raises:
        SchemaError: if validation finds any blocking problem.
    """
    df = read_raw_csv(path)
    report = validate_schema(df)
    if not report.is_valid:
        raise SchemaError("; ".join(report.errors))
    return cast_to_schema(df)


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------
@dataclass
class DuplicateReport:
    """Counts describing exact and feature-level duplicates in the full dataset."""

    n_rows: int
    n_unique_rows: int
    n_exact_duplicates: int  # rows identical (all 22 columns) to an earlier row
    n_rows_in_duplicate_groups: int  # rows whose full profile occurs more than once
    n_duplicate_groups: int
    largest_group_size: int
    exact_duplicates_by_class: dict[int, int]
    positive_rate_all: float
    positive_rate_after_dedup: float
    positive_rate_among_extra_copies: float
    n_feature_duplicates: int  # rows whose 21 features match an earlier row
    n_conflicting_profiles: int  # feature profiles that occur with both labels
    n_rows_in_conflicting_profiles: int
    expected_test_rows_with_train_copy: float  # under a random 80/20 split

    def summary_table(self) -> pd.DataFrame:
        """Two-column table (metric, value) for display in the notebook."""
        rows = asdict(self)
        rows["exact_duplicates_by_class"] = ", ".join(
            f"class {k}: {v:,}" for k, v in self.exact_duplicates_by_class.items()
        )
        return pd.DataFrame({"value": rows}).rename_axis("metric")


def duplicate_report(df: pd.DataFrame, test_size: float = config.TEST_SIZE) -> DuplicateReport:
    """Count exact duplicates, feature-only duplicates and label conflicts.

    ``expected_test_rows_with_train_copy`` is computed analytically, without
    performing any split: a row belonging to a group of ``k`` identical rows
    lands in the test set with probability ``test_size`` and has at least one
    copy in the training set with probability ``1 - test_size ** (k - 1)``.
    """
    cols = config.RAW_COLUMNS
    y = df[config.TARGET]

    exact_dup_mask = df.duplicated(subset=cols, keep="first")
    group_id = df.groupby(cols, sort=False).ngroup()
    group_sizes = group_id.value_counts()
    row_group_size = group_id.map(group_sizes).to_numpy()

    feature_dup_mask = df.duplicated(subset=config.FEATURES, keep="first")
    labels_per_profile = df.groupby(config.FEATURES, sort=False)[config.TARGET].nunique()
    conflicting = labels_per_profile[labels_per_profile > 1]
    rows_per_profile = df.groupby(config.FEATURES, sort=False)[config.TARGET].size()

    dedup = df.loc[~exact_dup_mask]
    extra = df.loc[exact_dup_mask, config.TARGET]
    p_copy_in_train = 1.0 - test_size ** (row_group_size - 1)
    expected_test_with_copy = float(test_size * p_copy_in_train.sum())

    return DuplicateReport(
        n_rows=len(df),
        n_unique_rows=int(len(dedup)),
        n_exact_duplicates=int(exact_dup_mask.sum()),
        n_rows_in_duplicate_groups=int((row_group_size > 1).sum()),
        n_duplicate_groups=int((group_sizes > 1).sum()),
        largest_group_size=int(group_sizes.max()),
        exact_duplicates_by_class={int(k): int(v) for k, v in extra.value_counts().sort_index().items()},
        positive_rate_all=float(y.mean()),
        positive_rate_after_dedup=float(dedup[config.TARGET].mean()),
        positive_rate_among_extra_copies=float(extra.mean()) if len(extra) else float("nan"),
        n_feature_duplicates=int(feature_dup_mask.sum()),
        n_conflicting_profiles=int(len(conflicting)),
        n_rows_in_conflicting_profiles=int(rows_per_profile.loc[conflicting.index].sum()),
        expected_test_rows_with_train_copy=expected_test_with_copy,
    )


def largest_duplicate_groups(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the ``n`` most frequent repeated full-row profiles with their counts."""
    counts = df.value_counts(subset=config.RAW_COLUMNS, sort=True).rename("count")
    return counts[counts > 1].head(n).reset_index()


# ---------------------------------------------------------------------------
# Train/test split (run once, in Phase 0)
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    """Indices (labels of the input frame) of the train and test partitions."""

    train_index: np.ndarray
    test_index: np.ndarray
    train_positive_rate: float
    test_positive_rate: float

    @property
    def class_ratio_gap(self) -> float:
        """Absolute difference between the train and test positive rates."""
        return abs(self.train_positive_rate - self.test_positive_rate)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> SplitResult:
    """Stratified train/test split on the target.

    Returns index labels rather than frames so the caller can save the test
    indices and the split can be audited against the raw file.
    """
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        stratify=df[config.TARGET].to_numpy(),
        random_state=seed,
        shuffle=True,
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    return SplitResult(
        train_index=train_idx,
        test_index=test_idx,
        train_positive_rate=float(df.loc[train_idx, config.TARGET].mean()),
        test_positive_rate=float(df.loc[test_idx, config.TARGET].mean()),
    )


def save_split(
    df: pd.DataFrame,
    split: SplitResult,
    manifest_extra: dict | None = None,
    processed_dir: Path = config.PROCESSED_DIR,
    overwrite: bool = False,
) -> dict:
    """Write train.csv, test.csv, test_indices.csv and split_manifest.json.

    The ``raw_index`` column in each file is the 0-based row position in the
    raw CSV, so every saved row can be traced back to its source.

    Raises:
        FileExistsError: if a split already exists and ``overwrite`` is False.
            The test set is created once; re-splitting needs an explicit
            decision.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_path = processed_dir / config.TEST_FILE
    if test_path.exists() and not overwrite:
        raise FileExistsError(
            f"{test_path} already exists. The test split is created once; "
            "pass overwrite=True only if the user has asked to re-split."
        )
    if np.intersect1d(split.train_index, split.test_index).size:
        raise ValueError("train and test indices overlap")

    train = df.loc[split.train_index].rename_axis("raw_index")
    test = df.loc[split.test_index].rename_axis("raw_index")
    train.to_csv(processed_dir / config.TRAIN_FILE)
    test.to_csv(test_path)
    pd.DataFrame({"raw_index": split.test_index}).to_csv(
        processed_dir / config.TEST_INDEX_FILE, index=False
    )

    manifest = {
        "seed": config.SEED,
        "test_size": config.TEST_SIZE,
        "n_input_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_positive_rate": split.train_positive_rate,
        "test_positive_rate": split.test_positive_rate,
        "class_ratio_gap": split.class_ratio_gap,
        "train_sha256": file_sha256(processed_dir / config.TRAIN_FILE),
        "test_sha256": file_sha256(test_path),
        **(manifest_extra or {}),
    }
    with open(processed_dir / config.SPLIT_MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_train(processed_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    """Load the training split (indexed by ``raw_index``) with int64 columns."""
    path = Path(processed_dir) / config.TRAIN_FILE
    return pd.read_csv(path, index_col="raw_index").astype(np.int64)
