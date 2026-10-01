"""TEST_LOCK: the test set can be evaluated once; a second call needs force=True."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import TestSetLockedError, evaluate_on_test
from src.features import save_split, stratified_split


@pytest.fixture
def split_dir(tmp_path, synthetic_df):
    save_split(synthetic_df, stratified_split(synthetic_df), processed_dir=tmp_path)
    return tmp_path


def _constant_predictor(value: float):
    return lambda X: np.full(len(X), value)


def test_first_call_evaluates_and_writes_lock(split_dir):
    lock = split_dir / config.TEST_LOCK_FILE
    assert not lock.exists()

    results = evaluate_on_test(
        {"const": _constant_predictor(0.3)}, {"const": 0.5}, processed_dir=split_dir
    )

    assert lock.exists()
    history = json.loads(lock.read_text(encoding="utf-8"))["history"]
    assert len(history) == 1 and history[0]["models"] == ["const"]
    assert history[0]["forced"] is False
    assert results.metrics.loc["const", "recall"] == 0.0
    assert len(results.y_true) == len(results.X_test) == len(results.probabilities["const"])
    assert list(results.X_test.columns) == config.FEATURES


def test_second_call_raises(split_dir):
    evaluate_on_test({"m": _constant_predictor(0.3)}, {"m": 0.5}, processed_dir=split_dir)
    with pytest.raises(TestSetLockedError):
        evaluate_on_test({"m": _constant_predictor(0.3)}, {"m": 0.5}, processed_dir=split_dir)


def test_force_overrides_lock_and_records_history(split_dir):
    evaluate_on_test({"m": _constant_predictor(0.3)}, {"m": 0.5}, processed_dir=split_dir)
    evaluate_on_test({"m": _constant_predictor(0.7)}, {"m": 0.5}, force=True, processed_dir=split_dir)

    history = json.loads((split_dir / config.TEST_LOCK_FILE).read_text(encoding="utf-8"))["history"]
    assert [entry["forced"] for entry in history] == [False, True]


def test_mismatched_names_do_not_burn_lock(split_dir):
    with pytest.raises(ValueError):
        evaluate_on_test({"a": _constant_predictor(0.3)}, {"b": 0.5}, processed_dir=split_dir)
    assert not (split_dir / config.TEST_LOCK_FILE).exists()


def test_invalid_probabilities_rejected(split_dir):
    with pytest.raises(ValueError):
        evaluate_on_test({"bad": _constant_predictor(1.5)}, {"bad": 0.5}, processed_dir=split_dir)


def test_metrics_match_hand_computation(split_dir):
    test = pd.read_csv(split_dir / config.TEST_FILE, index_col="raw_index")
    y = test[config.TARGET].to_numpy()

    # A perfect predictor: every metric at its best value.
    oracle = {"oracle": lambda X: y.astype(float)}
    m = evaluate_on_test(oracle, {"oracle": 0.5}, processed_dir=split_dir).metrics.loc["oracle"]
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        assert m[metric] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)
    assert m["tp"] == y.sum() and m["fn"] == 0


def test_real_project_lock_untouched_by_tests():
    """The suite only writes locks under tmp_path, never in data/processed/."""
    lock = config.PROCESSED_DIR / config.TEST_LOCK_FILE
    if lock.exists():
        pytest.skip("the real test set has already been evaluated (Phase 6 done)")
    assert not lock.exists()
