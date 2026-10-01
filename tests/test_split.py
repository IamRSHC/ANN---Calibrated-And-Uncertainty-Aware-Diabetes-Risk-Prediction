"""Stratified split: disjoint, complete, reproducible, class ratio preserved."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import load_train, save_split, stratified_split


def test_split_is_disjoint_and_complete(synthetic_df):
    split = stratified_split(synthetic_df)
    assert np.intersect1d(split.train_index, split.test_index).size == 0
    union = np.union1d(split.train_index, split.test_index)
    assert np.array_equal(union, synthetic_df.index.to_numpy())


def test_split_sizes(synthetic_df):
    split = stratified_split(synthetic_df)
    assert len(split.test_index) == pytest.approx(config.TEST_SIZE * len(synthetic_df), abs=1)


def test_class_ratio_within_gate(synthetic_df):
    split = stratified_split(synthetic_df)
    assert split.class_ratio_gap <= config.MAX_CLASS_RATIO_GAP


def test_split_is_reproducible(synthetic_df):
    a = stratified_split(synthetic_df)
    b = stratified_split(synthetic_df)
    assert np.array_equal(a.test_index, b.test_index)
    c = stratified_split(synthetic_df, seed=config.SEED + 1)
    assert not np.array_equal(a.test_index, c.test_index)


def test_save_split_round_trip(tmp_path, synthetic_df):
    split = stratified_split(synthetic_df)
    manifest = save_split(synthetic_df, split, {"duplicate_policy": "test"}, processed_dir=tmp_path)

    train = load_train(tmp_path)
    pd.testing.assert_frame_equal(
        train, synthetic_df.loc[split.train_index].rename_axis("raw_index")
    )
    saved_idx = pd.read_csv(tmp_path / config.TEST_INDEX_FILE)["raw_index"].to_numpy()
    assert np.array_equal(saved_idx, split.test_index)

    on_disk = json.loads((tmp_path / config.SPLIT_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["n_train"] + on_disk["n_test"] == len(synthetic_df)
    assert on_disk["duplicate_policy"] == "test"


def test_save_split_refuses_to_overwrite(tmp_path, synthetic_df):
    split = stratified_split(synthetic_df)
    save_split(synthetic_df, split, processed_dir=tmp_path)
    with pytest.raises(FileExistsError):
        save_split(synthetic_df, split, processed_dir=tmp_path)
