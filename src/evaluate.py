"""Metrics and the one-shot, locked evaluation on the held-out test set.

``evaluate_on_test`` is the only code path in the project that reads the test
split. The first call writes ``data/processed/TEST_LOCK``; any later call
raises :class:`TestSetLockedError` unless ``force=True`` is passed, and only
the user may pass that.
"""
from __future__ import annotations

import getpass
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

# A predictor maps raw feature rows (the 21 BRFSS columns) to P(y = 1).
Predictor = Callable[[pd.DataFrame], np.ndarray]


class TestSetLockedError(RuntimeError):
    """Raised when the test set has already been used for evaluation."""

    __test__ = False  # not a pytest test class


@dataclass
class TestResults:
    """Everything produced by the single test-set evaluation."""

    __test__ = False  # not a pytest test class

    metrics: pd.DataFrame  # one row per model
    y_true: np.ndarray
    probabilities: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    X_test: pd.DataFrame  # raw features, for bias audit and error analysis


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Threshold-dependent and threshold-free metrics for one model.

    Args:
        y_true: ground-truth labels (0/1).
        y_prob: predicted probability of the positive class.
        threshold: predict 1 when ``y_prob >= threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _read_lock(lock_path: Path) -> dict:
    with open(lock_path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_lock(lock_path: Path, model_names: list[str], thresholds: Mapping[str, float], force: bool) -> None:
    """Record this evaluation call. Earlier calls are kept in ``history``."""
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "models": model_names,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "forced": force,
    }
    history = _read_lock(lock_path)["history"] if lock_path.exists() else []
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        json.dump({"history": history + [entry]}, fh, indent=2)


def _read_test_split(processed_dir: Path) -> pd.DataFrame:
    """Read the test split. Called only from :func:`evaluate_on_test`."""
    path = Path(processed_dir) / config.TEST_FILE
    return pd.read_csv(path, index_col="raw_index").astype(np.int64)


def evaluate_on_test(
    predictors: Mapping[str, Predictor],
    thresholds: Mapping[str, float],
    *,
    force: bool = False,
    processed_dir: Path = config.PROCESSED_DIR,
) -> TestResults:
    """Evaluate every model on the test set exactly once.

    Each predictor receives the raw 21-feature test frame and must apply its
    own saved preprocessing pipeline. Thresholds must already be fixed (from
    out-of-fold or validation predictions); nothing is tuned here.

    Args:
        predictors: model name -> function returning P(y = 1) for each row.
        thresholds: model name -> decision threshold; keys must match ``predictors``.
        force: bypass an existing ``TEST_LOCK``. Only the user may set this.
        processed_dir: folder holding the split files and the lock.

    Raises:
        TestSetLockedError: if the test set was already evaluated and ``force`` is False.
        ValueError: if model names in ``predictors`` and ``thresholds`` differ,
            or a predictor returns probabilities of the wrong shape or range.
    """
    processed_dir = Path(processed_dir)
    lock_path = processed_dir / config.TEST_LOCK_FILE
    if lock_path.exists() and not force:
        previous = _read_lock(lock_path)["history"][-1]
        raise TestSetLockedError(
            f"The test set was already evaluated on {previous['timestamp_utc']} "
            f"(models: {previous['models']}). A second evaluation needs force=True, "
            "which only the user may pass."
        )

    # Validate inputs before touching the test file, so a typo does not burn the lock.
    if set(predictors) != set(thresholds):
        raise ValueError(
            f"predictors and thresholds name different models: "
            f"{sorted(set(predictors) ^ set(thresholds))}"
        )
    if not predictors:
        raise ValueError("no predictors given")

    model_names = list(predictors)
    _write_lock(lock_path, model_names, thresholds, force)

    test = _read_test_split(processed_dir)
    X_test = test[config.FEATURES]
    y_true = test[config.TARGET].to_numpy()

    probabilities: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name in model_names:
        prob = np.asarray(predictors[name](X_test), dtype=np.float64).reshape(-1)
        if prob.shape != y_true.shape:
            raise ValueError(f"{name}: expected {y_true.shape[0]} probabilities, got {prob.shape}")
        if np.any(~np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
            raise ValueError(f"{name}: probabilities must be finite and within [0, 1]")
        probabilities[name] = prob
        predictions[name] = (prob >= thresholds[name]).astype(int)
        rows.append({"model": name, **classification_metrics(y_true, prob, thresholds[name])})

    return TestResults(
        metrics=pd.DataFrame(rows).set_index("model"),
        y_true=y_true,
        probabilities=probabilities,
        predictions=predictions,
        X_test=X_test,
    )
