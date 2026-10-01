"""Schema validation and duplicate counting on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import SchemaError, cast_to_schema, duplicate_report, load_raw, validate_schema


def test_valid_frame_passes(synthetic_df):
    report = validate_schema(synthetic_df.astype(float))  # raw CSV stores floats
    assert report.is_valid, report.errors
    assert report.columns_match and report.column_order_matches
    assert cast_to_schema(synthetic_df.astype(float)).dtypes.eq(np.int64).all()


@pytest.mark.parametrize(
    "col, bad_value, fragment",
    [
        ("HighBP", 2, "outside"),
        ("BMI", 5, "outside"),
        ("MentHlth", 31, "outside"),
        ("Age", 0, "outside"),
        ("Income", 3.5, "non-integer"),
        ("GenHlth", np.nan, "missing"),
    ],
)
def test_bad_values_are_reported(synthetic_df, col, bad_value, fragment):
    df = synthetic_df.astype(float)
    df.loc[0, col] = bad_value
    report = validate_schema(df)
    assert not report.is_valid
    assert any(col in e and fragment in e for e in report.errors)


def test_missing_and_extra_columns(synthetic_df):
    df = synthetic_df.drop(columns=["Sex"]).assign(Extra=1)
    report = validate_schema(df)
    assert report.missing_columns == ["Sex"]
    assert report.unexpected_columns == ["Extra"]
    assert not report.is_valid


def test_load_raw_rejects_invalid_file(tmp_path, synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "BMI"] = 200
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SchemaError):
        load_raw(path)


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "absent.csv")


def test_duplicate_counts():
    base = pd.DataFrame(
        {col: [config.VALUE_RANGES[col][0]] * 6 for col in config.RAW_COLUMNS}
    )
    base[config.TARGET] = [0, 0, 0, 1, 1, 0]
    base.loc[5, "BMI"] = 30  # one unique row
    # Rows 0-2: identical, label 0. Rows 3-4: same features, label 1. Row 5: unique.
    report = duplicate_report(base, test_size=0.2)

    assert report.n_exact_duplicates == 3  # rows 1, 2 and 4
    assert report.n_unique_rows == 3
    assert report.n_duplicate_groups == 2
    assert report.n_rows_in_duplicate_groups == 5
    assert report.largest_group_size == 3
    assert report.exact_duplicates_by_class == {0: 2, 1: 1}
    assert report.n_feature_duplicates == 4  # rows 1-4 share row 0's features
    assert report.n_conflicting_profiles == 1
    assert report.n_rows_in_conflicting_profiles == 5
    # 3 rows in a group of 3: 0.2 * (1 - 0.2**2) each; 2 rows in a group of 2: 0.2 * (1 - 0.2) each.
    expected = 3 * 0.2 * (1 - 0.2**2) + 2 * 0.2 * (1 - 0.2)
    assert report.expected_test_rows_with_train_copy == pytest.approx(expected)
