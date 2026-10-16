"""Model logic shared by the Streamlit app and the tests: load the NumPy ANN, preprocess one respondent, MC-dropout risk."""
import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(model_dir: Path = MODEL_DIR):
    cfg = json.loads((model_dir / "proposed_ann_config.json").read_text())
    w = np.load(model_dir / "proposed_ann.npz")
    params = [w[f"arr_{i}"] for i in range(len(w.files))]
    return cfg, params


def preprocess(row: dict, cfg: dict) -> np.ndarray:
    x = []
    for f in cfg["features"]:
        v = float(row[f])
        if f == "BMI":
            v = min(max(v, cfg["bmi_clip"][0]), cfg["bmi_clip"][1])
        if f in cfg["continuous"]:
            v = (v - cfg["mu"][f]) / cfg["sd"][f]
        elif f in cfg["ordinal"]:
            v = (v - cfg["ordinal_min"][f]) / (cfg["ordinal_max"][f] - cfg["ordinal_min"][f])
        x.append(v)
    return np.array(x)[None, :]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mc_risk(X: np.ndarray, cfg: dict, params: list, passes: int | None = None, seed: int = 42):
    """MC-dropout forward passes with calibration (temperature + bias). Returns (mean risk, std) per row."""
    passes = passes or cfg["mc_passes"]
    rng, rate = np.random.default_rng(seed), cfg["dropout"]
    W = params[0::2]; b = params[1::2]
    out = []
    for _ in range(passes):
        h = X
        for k in range(len(W) - 1):
            h = np.maximum(h @ W[k] + b[k], 0)
            h = h * (rng.random(h.shape) >= rate) / (1 - rate)
        out.append(sigmoid((h @ W[-1] + b[-1])[:, 0] / cfg["temperature"] + cfg.get("bias", 0.0)))
    P = np.stack(out)
    return P.mean(0), P.std(0)


def preprocess_frame(df, cfg: dict) -> np.ndarray:
    """Vectorised version of preprocess() for many respondents (a DataFrame with the 21 feature columns)."""
    return np.vstack([preprocess(r, cfg) for r in df[cfg["features"]].to_dict("records")])


def decision_margin(risk, unc, cfg: dict):
    """MC standard deviations between the risk and the decision threshold; refer to doctor when below tau."""
    return np.abs(np.asarray(risk) - cfg["threshold"]) / (np.asarray(unc) + 1e-9)
