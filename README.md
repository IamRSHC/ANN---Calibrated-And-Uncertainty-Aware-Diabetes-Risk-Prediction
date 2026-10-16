# Diabetes Disease Prediction System

21CSE326T Artificial Neural Networks, SRMIST Tiruchirappalli (B.Tech CSE AI & ML, Sem V, 2026–27).

This project compares classical ML/ANN classifiers and a stacking ensemble against a neural network written from scratch in NumPy. The NumPy model adds uncertainty estimation, calibration, explainability and what-if analysis. It is trained on the CDC BRFSS 2015 Diabetes Health Indicators dataset.

## Setup

Requires Python 3.12.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the Kaggle dataset
"Diabetes Health Indicators Dataset" (CDC BRFSS 2015) and place it in `data/raw/`.
The raw file is gitignored; the train/test split in `data/processed/` is committed.

## Tests

```bash
pytest
```

## Test-set rule

The test split is created once (stratified 80/20, seed 42) and is read only by
`src/evaluate.py::evaluate_on_test()`. The first call writes `data/processed/TEST_LOCK`,
and every later call fails unless `force=True` is passed.

## Review II

| Path | What it is |
|---|---|
| `notebooks/Diabetes_ANN_Project.ipynb` | Complete project, executed with outputs (Model 1, NumPy ANN, ablation, locked-test evaluation, uncertainty, SHAP, bias audit) |
| `app/app.py`, `app/model.py` | Streamlit screening prototype; `model.py` holds the NumPy forward pass shared with the tests |
| `models/` | Trained ANN weights and preprocessing / calibration config (written by the notebook) |
| `reports/figures/`, `reports/results/` | All figures and result tables from the notebook |
| `tests/test_review2.py` | Checks on the saved model, preprocessing, risk ordering, focal-loss gradient and a real-data sample |

```bash
pytest tests/test_review2.py -v      # model checks (about 10 s)
streamlit run app/app.py             # opens the app at http://localhost:8501
```

Re-running the notebook (Kernel -> Restart & Run All, about 10-20 min on a laptop CPU) regenerates `models/` and `reports/`.
