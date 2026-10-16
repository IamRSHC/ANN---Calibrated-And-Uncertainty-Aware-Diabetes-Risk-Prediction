# Calibrated and Uncertainty-Aware Diabetes Risk Prediction Using ANNs

21CSE326T Artificial Neural Networks, SRMIST Tiruchirappalli (B.Tech CSE AI & ML, Sem V, 2026–27).
Team: Hareecharan R S, V Shakthi Balan and Manoranjan U S.

The project compares two approaches on the CDC BRFSS 2015 Diabetes Health Indicators survey (253,680 respondents, 21 self-reported indicators, 13.93% diabetic):

- **Model 1 (existing algorithms):** Logistic Regression, KNN, SVM (RBF), Random Forest, XGBoost, Gaussian Naive Bayes, an MLP, and a stacking ensemble built on top of them.
- **Model 2 (built from scratch):** a 21-64-32-1 neural network written entirely in NumPy. The forward pass, backpropagation, dropout and Adam are all hand-written, and the backpropagation is verified by a numerical gradient check.

Model 2 adds four novelty components:

1. A class-weighted focal loss with a hand-derived gradient.
2. Monte Carlo dropout uncertainty (50 stochastic passes).
3. Calibration by temperature scaling with a bias term.
4. A "refer to doctor" abstention rule based on the decision margin |μ − t| / σ.

## Results on the locked test set (50,736 respondents)

| Model | ROC-AUC | PR-AUC | Recall | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.824 | 0.412 | 0.834 | 0.177 |
| XGBoost | 0.830 | 0.426 | 0.823 | 0.176 |
| Stacking ensemble | 0.830 | 0.425 | 0.845 | 0.097 |
| **Proposed ANN (NumPy)** | **0.828** | **0.421** | **0.838** | **0.097** |

- **Ranking:** the proposed ANN is level with the best baselines (bootstrap 95% confidence intervals overlap).
- **Calibration:** its risk estimates are calibrated (Brier 0.097 vs 0.176 for XGBoost).
- **Abstention:** the 10% of respondents it refers to a doctor have an error rate of 49.9%, against 29.1% for the cases it decides.

All nine models, all metrics and all significance tests are in the notebook.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/Diabetes_ANN_Project.ipynb` | **The complete project**, executed with all outputs. It covers data validation, the group-aware split, preprocessing, Model 1, Model 2, the ablation study, the one-time test evaluation (classification reports, confusion matrices, ROC / PR / calibration curves, McNemar, bootstrap CIs), uncertainty analysis, SHAP and what-if explanations, and a bias audit. |
| `app/app.py`, `app/model.py` | Streamlit screening prototype for Model 2. It shows calibrated risk, an uncertainty band, the refer-to-doctor flag, risk drivers and what-if sliders. `model.py` holds the NumPy forward pass shared with the tests. |
| `models/` | Trained Model 2 weights and its preprocessing / calibration config, written by the notebook. |
| `reports/figures/`, `reports/results/` | Every figure and results table produced by the notebook. |
| `tests/test_review2.py` | Checks on the saved model, preprocessing, risk ordering, MC dropout, the abstention rule, the focal-loss gradient and a real-data sample. |
| `src/`, `tests/test_*.py`, `notebooks/00_setup_split.ipynb` | Phase 0 utilities: schema validation, a split helper and a test-set lock, with their tests. |

## Setup

Requires Python 3.12.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`app/requirements.txt` lists only the three packages the app needs (numpy, pandas, streamlit). It is used for deployment.

## Data

Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the Kaggle dataset "Diabetes Health Indicators Dataset" (CDC BRFSS 2015, A. Teboul) and place it in `data/raw/`. The raw file is not committed.

The app and the model tests do not need the data. Only the notebook and the real-data test use it.

## Running

```bash
pytest                               # all tests (the real-data test is skipped if data/raw/ is empty)
streamlit run app/app.py             # app at http://localhost:8501
```

Open the notebook in VS Code or Jupyter to read it; all outputs are saved in the file. Running it again (Restart & Run All, about 10–20 minutes on a laptop CPU) regenerates `models/` and `reports/`.

## Evaluation protocol

- **Split:** one group-aware stratified 80/20 split (seed 42). The 24,206 exact duplicate rows are grouped so that no test respondent has an identical copy in the training data. Both splits keep the real 13.93% prevalence.
- **Fitting:** a validation split (also group-aware) is carved from the training data. It is used for early stopping, calibration, decision thresholds (F2-optimal), the abstention threshold and the stacking meta-learner.
- **Test set:** used once, at the end of the notebook, after every parameter has been fixed.

## Disclaimer

This is an academic screening prototype trained on self-reported 2015 US survey data. It is not a diagnostic tool. The model learns associations, not causes: for example, heavy drinking appears to lower risk because of reverse causation in the survey data.
