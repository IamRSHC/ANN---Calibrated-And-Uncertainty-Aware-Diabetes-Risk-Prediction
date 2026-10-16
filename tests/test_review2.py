"""Review II checks: saved ANN model, app preprocessing, risk ordering, focal-loss gradient, and real-data sanity.

Run from the project root:  pytest tests/test_review2.py -v
The real-data test is skipped automatically if data/raw/ does not contain the binary CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from model import decision_margin, load_model, mc_risk, preprocess, preprocess_frame, sigmoid  # noqa: E402

RAW = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"

HEALTHY = {"HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
           "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
           "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 3, "Education": 6, "Income": 8}
HIGH_RISK = dict(HEALTHY, HighBP=1, HighChol=1, BMI=40, PhysActivity=0, GenHlth=4, PhysHlth=15, DiffWalk=1,
                 HeartDiseaseorAttack=1, Age=11, Income=3, Education=4)


@pytest.fixture(scope="module")
def model():
    if not (ROOT / "models" / "proposed_ann.npz").exists():
        pytest.skip("models/proposed_ann.npz missing - run the notebook first")
    return load_model()


def test_saved_weights_have_the_21_64_32_1_architecture(model):
    cfg, params = model
    assert [p.shape for p in params] == [(21, 64), (64,), (64, 32), (32,), (32, 1), (1,)]
    assert len(cfg["features"]) == 21
    assert all(np.isfinite(p).all() for p in params)


def test_config_holds_calibration_and_abstention_settings(model):
    cfg, _ = model
    for key in ("temperature", "bias", "threshold", "tau", "mc_passes", "dropout"):
        assert key in cfg, key
    assert cfg["temperature"] > 0 and 0 < cfg["threshold"] < 1 and cfg["tau"] > 0 and cfg["mc_passes"] == 50


def test_preprocess_clips_bmi_and_scales_ordinals(model):
    cfg, _ = model
    lo, hi = cfg["bmi_clip"]
    i = cfg["features"].index("BMI")
    x_max = preprocess(dict(HEALTHY, BMI=98), cfg)[0, i]
    x_hi = preprocess(dict(HEALTHY, BMI=hi), cfg)[0, i]
    assert x_max == pytest.approx(x_hi), "BMI above the 99th percentile must be clipped"
    x = preprocess(HIGH_RISK, cfg)[0]
    for f in cfg["ordinal"]:
        assert 0.0 <= x[cfg["features"].index(f)] <= 1.0, f


def test_high_risk_profile_scores_above_healthy_profile(model):
    cfg, params = model
    r_low, _ = mc_risk(preprocess(HEALTHY, cfg), cfg, params)
    r_high, _ = mc_risk(preprocess(HIGH_RISK, cfg), cfg, params)
    assert r_low[0] < cfg["threshold"] < r_high[0]
    assert r_high[0] - r_low[0] > 0.3


def test_mc_dropout_is_reproducible_and_has_spread(model):
    cfg, params = model
    x = preprocess(HIGH_RISK, cfg)
    a, sa = mc_risk(x, cfg, params, seed=7)
    b, sb = mc_risk(x, cfg, params, seed=7)
    assert a[0] == b[0] and sa[0] == sb[0]
    assert sa[0] > 0, "dropout must stay active at inference"


def test_decision_margin_refers_only_borderline_cases(model):
    cfg, _ = model
    t = cfg["threshold"]
    assert decision_margin(t, 0.05, cfg) < cfg["tau"]          # right on the threshold -> refer
    assert decision_margin(t + 0.5, 0.05, cfg) > cfg["tau"]    # far above -> model decides


def test_focal_loss_gradient_matches_numerical_derivative():
    alpha, gamma, eps = 0.86, 2.0, 1e-6

    def fl(z, y):
        p = sigmoid(z)
        return -alpha * (1 - p) ** gamma * np.log(p) if y == 1 else -(1 - alpha) * p ** gamma * np.log(1 - p)

    def grad(z, y):
        p = sigmoid(z)
        if y == 1:
            return alpha * (1 - p) ** gamma * (gamma * p * np.log(p) - (1 - p))
        return (1 - alpha) * p ** gamma * (p - gamma * (1 - p) * np.log(1 - p))

    for z in (-3.0, -0.5, 0.0, 0.7, 2.5):
        for y in (0, 1):
            num = (fl(z + eps, y) - fl(z - eps, y)) / (2 * eps)
            assert grad(z, y) == pytest.approx(num, rel=1e-5, abs=1e-9)


@pytest.mark.skipif(not RAW.exists(), reason="binary BRFSS CSV not in data/raw/")
def test_real_data_sample_is_calibrated_and_discriminates(model):
    from sklearn.metrics import roc_auc_score
    cfg, params = model
    df = pd.read_csv(RAW).sample(5000, random_state=0)
    mu, _ = mc_risk(preprocess_frame(df, cfg), cfg, params)
    y = df["Diabetes_binary"].to_numpy()
    assert abs(mu.mean() - y.mean()) < 0.02, f"mean risk {mu.mean():.3f} vs rate {y.mean():.3f}"
    assert roc_auc_score(y, mu) > 0.80
