# CLAUDE.md — Diabetes Disease Prediction System

## 1. Project context
- **Course:** 21CSE326T Artificial Neural Networks, SRMIST Tiruchirappalli, B.Tech CSE (AI & ML), Sem V, AY 2026–27
- **Project:** Diabetes Disease Prediction System. Team project, built end to end by R S Hareecharan (GitHub: IamRSHC).
- **Goal:** Compare existing ML/ANN classifiers (plus a stacking ensemble) against a proposed model built from scratch in NumPy, with a novelty framework (uncertainty, calibration, explainability, what-if analysis).
- **Dataset:** CDC BRFSS 2015 Diabetes Health Indicators (approved by faculty).

### Review schedule (the deadlines that matter)
| Review | Date | Marks | Covers |
|---|---|---|---|
| I | 08 Oct 2026 | 40 | Problem definition (5), literature survey & gap (8), dataset & data understanding (8), preprocessing (5), EDA (7), feature engineering & selection (7) |
| II | 15 Oct 2026 | 40 | Implementation (10), novelty / proposed framework (10), evaluation (8), explainability (8), application / UI (4) |
| III | 22 Oct 2026 | 20 | Documentation & presentation (15), viva & individual contribution (5) |

## 2. Hard rules (never break these)

### 2.1 Test-set rule
- The test set is created **once** in Phase 0 (stratified 80/20, SEED=42) and saved to `data/processed/`.
- It must not be loaded, inspected, or used in any way before Phase 6.
- All test evaluation goes through `src/evaluate.py::evaluate_on_test()`, which writes `data/processed/TEST_LOCK`. A second call raises an error unless `force=True` is passed, and only the user may pass that.
- Never retune, re-select features, or change thresholds after looking at test results.

### 2.2 Leakage rules
- Every fitted transformation (clipping bounds, scalers, encoders, feature selection, SMOTENC/undersampling, SOM, temperature scaling, decision thresholds) is fit on **training data or training folds only**.
- Resampling only happens inside `imblearn.pipeline.Pipeline`, so it runs per CV fold.
- EDA runs on the **training split only**.
- Imputation must never use the target (no class-conditional imputation).

### 2.3 Shared preprocessing
- All preprocessing and feature engineering lives in `src/features.py`.
- Notebooks and the Streamlit app both import from it.
- The fitted pipeline is saved alongside each model; the app loads and applies it.
- The app must never re-implement any transformation.

### 2.4 Reproducibility
- `SEED = 42` is set in `src/config.py` and passed to every random component (NumPy, sklearn, XGBoost, splits, CV, SOM, dropout).
- `requirements.txt` is pinned (exact versions) at Phase 0.
- Before any commit: notebooks must pass **Restart Kernel & Run All** with no errors.

### 2.5 Stop and ask the user before:
- deciding how to handle duplicate rows (Phase 0)
- changing the architecture, folder structure, dataset, or primary metric
- dropping any requirement or rubric item
- anything that deletes data or rewrites git history

## 3. Environment
- Python 3.12.x (required) in a virtual environment (`.venv/`, gitignored)
- Core libraries: numpy, pandas, scipy, scikit-learn, imbalanced-learn, xgboost, shap, statsmodels, matplotlib, seaborn, joblib, jupyter, streamlit
- **The from-scratch model (`src/scratch/`) uses NumPy only.** No sklearn, PyTorch, TensorFlow, or autograd inside it.
- The dataset is downloaded manually by the user into `data/raw/` (no API credentials are handled by Claude).

## 4. Dataset
- **File:** `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` (the binary, imbalanced version; expected ~253,680 rows — verify on load)
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes)
- **Features (21):**
  - Binary (0/1): HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
  - Continuous: BMI, MentHlth (0–30 days), PhysHlth (0–30 days)
  - Ordinal: GenHlth (1–5), Age (1–13, BRFSS age bands), Education (1–6), Income (1–8)
- **Known limitations** (to be documented):
  - self-reported survey data
  - no glucose, insulin, or HbA1c values
  - class imbalance (~14% positive)
  - many exact duplicate rows are possible, because 21 coarse features let distinct respondents produce identical rows

## 5. Folder structure
```
diabetes-ann/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore                # .venv/, data/raw/, notes/, __pycache__/, .ipynb_checkpoints/
├── data/
│   ├── raw/                  # gitignored
│   └── processed/            # train/test splits, test indices, TEST_LOCK
├── notebooks/
│   ├── 00_setup_split.ipynb
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_preprocessing_fe.ipynb
│   ├── 03_baselines_ensemble.ipynb
│   ├── 04_scratch_model.ipynb
│   └── 05_evaluation_xai.ipynb
├── src/
│   ├── config.py             # SEED, paths, constants
│   ├── features.py           # preprocessing + FE pipeline (shared with app)
│   ├── evaluate.py           # metrics, plots, McNemar, bootstrap CI, evaluate_on_test()
│   ├── xai.py                # SHAP + what-if helpers
│   └── scratch/
│       ├── layers.py         # Dense, ReLU, Sigmoid, Dropout
│       ├── losses.py         # BCE, class-weighted focal loss
│       ├── optim.py          # Adam
│       ├── model.py          # MLP, training loop, early stopping, MC dropout
│       ├── calibration.py    # temperature scaling
│       ├── som.py            # Self-Organizing Map (stretch goal)
│       └── gradcheck.py      # numerical gradient checking
├── tests/                    # pytest: gradient checks, pipeline leakage tests, lock test
├── models/                   # saved models + fitted pipelines (.joblib / .npz)
├── reports/
│   ├── figures/              # all plots (PNG, 300 dpi, labelled axes, titles)
│   └── results/              # CSV/JSON metric tables
├── app/
│   └── streamlit_app.py
└── notes/                    # gitignored; module explanations for viva prep
```

## 6. Coding standards
- Complete, working code only. No placeholders, no `TODO: implement`, no `pass` stubs.
- Logic lives in `src/`; notebooks import from `src/` and orchestrate.
- Type hints and docstrings on all public functions.
- Every figure is saved to `reports/figures/` with a descriptive filename; every metric table goes to `reports/results/`.
- After completing each module, write a plain explanation of it (what it does, why, the key math) to `notes/<module>.md` for viva preparation.
- Commit at each phase gate with the message format `phaseN: <what was done>`.

## 7. Phase plan

### Phase 0 — Setup & test-set lock (01 Oct)
- Repo, venv, `config.py`, `.gitignore`, pinned `requirements.txt`
- Load the CSV, validate the schema (22 columns, dtypes, value ranges per section 4), count exact duplicates → **STOP and report to the user; the user decides how duplicates are handled**
- Apply the user's duplicate decision, then do a stratified 80/20 split with SEED=42 and save the train, test, and test-index files
- **Gate:** class ratio within ±0.5% between splits; files committed; `TEST_LOCK` mechanism implemented and tested

### Phase 1 — Data understanding & EDA (02–03 Oct) → Review I
- Feature dictionary table (name, type, range, meaning)
- Data-quality report: missing/invalid values (document the check even if none are found), BMI outliers (IQR + percentile view), duplicates, value-range validation
- Visualizations (training split only):
  - class balance
  - distributions by outcome
  - correlation heatmap (Spearman)
  - diabetes rate by Age band × BMI category (heatmap)
  - diabetes rate by GenHlth, HighBP, HighChol, Income
- An interpretation cell after every plot, with the key numbers behind it
- **Gate:** all figures saved; every figure has an interpretation cell

### Phase 2 — Preprocessing, feature engineering & selection (04–05 Oct) → Review I
- `features.py` pipeline (sklearn `ColumnTransformer`):
  - BMI clipped at training-set 1st/99th percentiles
  - StandardScaler on BMI, MentHlth, PhysHlth
  - ordinal features kept as integer codes
  - binary features passed through
- Engineered features:
  - CardioMetabolicScore = HighBP + HighChol + HeartDiseaseorAttack + Stroke
  - LifestyleScore = PhysActivity + Fruits + Veggies − Smoker − HvyAlcoholConsump
  - HealthcareAccess = AnyHealthcare − NoDocbcCost
  - BMI category (WHO bands)
  - interaction terms BMI×Age and HighBP×HighChol
- Feature selection: mutual information, chi-square, RFE (on LR)
- Experimental comparison with 5-fold stratified CV, LR and XGBoost, three feature sets (raw / raw+engineered / selected); report PR-AUC, ROC-AUC, F1, recall as mean ± std
- Leakage test in `tests/`: pipeline statistics must come from training data only
- **Gate:** comparison table saved; final feature set chosen and justified with numbers

### Phase 3 — Imbalance strategy (05 Oct, can overlap Phase 2)
- Compare with LR, 5-fold stratified CV: no resampling / class weights / SMOTENC / random undersampling
- Plain SMOTE is NOT allowed: it creates fractional values in binary features
- **Gate:** strategy chosen per model family, with the result table saved

### Phase 4 — Baselines & ensemble (08–09 Oct) → Review II
- Seven baselines: Logistic Regression, KNN, SVM (RBF), Random Forest, XGBoost, Gaussian Naive Bayes, sklearn MLPClassifier
- KNN and SVM are trained on a stratified 25,000-row subsample of training data (state this in results)
- Tuning: RandomizedSearchCV, StratifiedKFold(5), scoring = average_precision (PR-AUC)
- Save out-of-fold (OOF) probabilities for every model
- Decision threshold per model chosen on OOF predictions by maximizing F2-score (recall-weighted, clinically justified)
- Stacking ensemble: StackingClassifier with an LR meta-learner over the best non-subsampled base models (selected by CV PR-AUC)
- SHAP: TreeExplainer on XGBoost (global summary + dependence plots)
- **Gate:** CV results table (mean ± std) saved; all models saved to `models/`

### Phase 5 — Proposed model from scratch (09–11 Oct) → Review II
**Architecture (starting point; tuned on a validation split carved from training data):**
- Input → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
- He initialization, Adam (lr 1e-3), mini-batch 512, early stopping on validation PR-AUC (patience 10), float64

**Steps with gates:**
| Step | Work | Gate |
|---|---|---|
| 5a | Layers, BCE loss, Adam, training loop | Gradient check (central difference, ε=1e-5): relative error < 1e-6 for every parameter |
| 5b | Train the plain MLP | Validation ROC-AUC within ~0.02 of LR; if clearly worse, debug before continuing |
| 5c | Class-weighted focal loss: FL = −α·y·(1−p)^γ·log(p) − (1−α)·(1−y)·p^γ·log(1−p), with hand-derived gradient | Gradient check passes; α and γ tuned on validation |
| 5d | MC dropout: T=50 stochastic passes → mean probability + std as uncertainty; abstention ("refer to doctor") when std exceeds a threshold chosen on validation | Accuracy-vs-coverage curve saved |
| 5e | Temperature scaling: fit T on validation logits by minimizing NLL | Reliability diagram + Brier score, before vs after |
| 5f (stretch) | SOM from scratch (10×10 grid, trained on training data only); BMU distance features appended to the MLP input | U-matrix saved; only if 5a–5e are done by end of 11 Oct |
| 5g | Ablation: plain MLP → +focal → +MC dropout/calibration → (+SOM) | Ablation table saved; shows each component's contribution |

### Phase 6 — Final evaluation on test set (12 Oct) → Review II
- Single call to `evaluate_on_test()` for all models
- Metrics per model at the tuned threshold: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, Brier score
- Plots:
  - confusion matrices
  - overlaid ROC curves
  - overlaid PR curves
  - calibration curves (all models)
- Statistics: bootstrap 95% CIs (1,000 resamples); McNemar test (statsmodels) between the proposed model and each baseline
- XAI: SHAP (global + local) for the best baseline; KernelSHAP or permutation importance on a ≤2,000-row sample for the proposed model; uncertainty interpretation on abstained cases
- Bias audit: recall and specificity by Sex, Age band, Income band
- False-negative analysis: profile of missed diabetic patients
- **Gate:** final comparison table + all figures saved; no retuning after this point

### Phase 7 — Application (13 Oct) → Review II
- Streamlit app loading the saved pipeline + proposed model (+ best baseline for comparison)
- Input validation:
  - BMI 12–98
  - MentHlth and PhysHlth 0–30
  - ordinal fields as dropdowns with human-readable labels (e.g. Age band "45–49")
  - all fields required
  - clear error messages
- Outputs:
  - calibrated risk %
  - uncertainty band
  - "refer to doctor" flag
  - local explanation (top contributing features)
- What-if analysis: sliders/toggles (BMI, HighBP, PhysActivity, etc.) showing how the risk changes
- Visible disclaimer: screening aid, not a medical diagnosis
- **Gate:** runs end to end with `streamlit run app/streamlit_app.py`; 5 manual test profiles give sensible outputs; invalid inputs are rejected cleanly

### Phase 8 — Final polish & reproducibility (16–21 Oct)
- Architecture diagram (generated figure)
- Final figures and results tables
- README with setup and run instructions
- Reproducibility check: fresh clone → install → run all notebooks → launch app
- Code freeze on 21 Oct

## 8. Definition of done (every phase)
- [ ] Notebooks pass Restart & Run All
- [ ] Figures and tables saved to `reports/`
- [ ] Tests pass (`pytest`)
- [ ] `notes/` updated with explanations for any new module
- [ ] Gate criteria met
- [ ] Committed with a `phaseN:` message
