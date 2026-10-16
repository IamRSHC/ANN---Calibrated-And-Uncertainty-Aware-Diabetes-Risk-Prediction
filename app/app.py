"""Streamlit app: calibrated, uncertainty-aware diabetes risk screening with the from-scratch NumPy ANN.

Run from the project root:  streamlit run app/app.py
Needs models/proposed_ann.npz and models/proposed_ann_config.json (written by the notebook, Section 12).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MODEL_DIR, decision_margin, load_model, mc_risk, preprocess  # noqa: E402

load_model = st.cache_resource(load_model)

AGE_BANDS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
EDUCATION = ["Never attended / kindergarten", "Grades 1-8", "Grades 9-11", "High-school graduate", "Some college", "College graduate"]
INCOME = ["< $10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k", "$35-50k", "$50-75k", "> $75k"]
GENHLTH = ["Excellent", "Very good", "Good", "Fair", "Poor"]
YES_NO = {
    "HighBP": "Told by a health professional you have high blood pressure?",
    "HighChol": "Told you have high cholesterol?",
    "CholCheck": "Cholesterol checked in the past 5 years?",
    "Smoker": "Smoked at least 100 cigarettes in your life?",
    "Stroke": "Ever had a stroke?",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack?",
    "PhysActivity": "Physical activity in the past 30 days (not job-related)?",
    "Fruits": "Eat fruit at least once a day?",
    "Veggies": "Eat vegetables at least once a day?",
    "HvyAlcoholConsump": "Heavy drinker (men > 14, women > 7 drinks/week)?",
    "AnyHealthcare": "Have any kind of health-care coverage?",
    "NoDocbcCost": "Could not see a doctor in the past year because of cost?",
    "DiffWalk": "Serious difficulty walking or climbing stairs?",
}
LABELS = {"BMI": "BMI", "MentHlth": "Poor mental-health days", "PhysHlth": "Poor physical-health days", "GenHlth": "General health",
          "Age": "Age band", "Education": "Education", "Income": "Income", "Sex": "Sex", **{k: k for k in YES_NO}}


st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")
if not (MODEL_DIR / "proposed_ann.npz").exists():
    st.error("Model files not found. Run the project notebook to the end first (it writes models/proposed_ann.npz).")
    st.stop()
cfg, params = load_model()

st.title("Diabetes Risk Screening")
st.caption("Calibrated, uncertainty-aware ANN trained on CDC BRFSS 2015 survey data (21 health indicators). "
           "Screening aid for education only - not a diagnosis.")

with st.sidebar:
    st.header("Your answers")
    row = {}
    c1, c2 = st.columns(2)
    height = c1.number_input("Height (cm)", 100.0, 230.0, 170.0, 0.5)
    weight = c2.number_input("Weight (kg)", 25.0, 300.0, 75.0, 0.5)
    bmi = weight / (height / 100) ** 2
    if not 12 <= bmi <= 98:
        st.error(f"BMI {bmi:.1f} is outside the valid survey range 12-98. Check height and weight.")
        st.stop()
    row["BMI"] = round(bmi)
    st.write(f"BMI: **{bmi:.1f}**")
    row["Sex"] = 1 if st.radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0
    row["Age"] = AGE_BANDS.index(st.selectbox("Age", AGE_BANDS, index=6)) + 1
    row["GenHlth"] = GENHLTH.index(st.selectbox("In general, your health is", GENHLTH, index=2)) + 1
    row["MentHlth"] = st.slider("Days of poor mental health (past 30)", 0, 30, 0)
    row["PhysHlth"] = st.slider("Days of poor physical health (past 30)", 0, 30, 0)
    row["Education"] = EDUCATION.index(st.selectbox("Education", EDUCATION, index=4)) + 1
    row["Income"] = INCOME.index(st.selectbox("Household income", INCOME, index=5)) + 1
    for k, q in YES_NO.items():
        default = 1 if k in ("CholCheck", "Fruits", "Veggies", "AnyHealthcare", "PhysActivity") else 0
        row[k] = 1 if st.radio(q, ["No", "Yes"], index=default, horizontal=True, key=k) == "Yes" else 0

mu, sd = mc_risk(preprocess(row, cfg), cfg, params)
risk, unc = float(mu[0]), float(sd[0])
margin = float(decision_margin(risk, unc, cfg))
refer = margin < cfg["tau"]
flag = risk >= cfg["threshold"]

a, b, c = st.columns(3)
a.metric("Calibrated diabetes risk", f"{risk:.1%}")
b.metric("Model uncertainty (σ of 50 passes)", f"{unc:.3f}", help=f"Decision margin |risk - threshold| / σ = {margin:.1f}; the case is referred when it is below τ = {cfg['tau']:.2f}")
c.metric("Screening result", "Refer to doctor" if refer else ("Higher risk" if flag else "Lower risk"))
lo, hi = max(0.0, risk - 2 * unc), min(1.0, risk + 2 * unc)
st.progress(min(risk, 1.0), text=f"Risk band (mean ± 2σ): {lo:.1%} - {hi:.1%}")

if refer:
    st.warning("The model is **not confident** about this decision: the risk lies too close to the screening threshold for its uncertainty. It abstains, so please consult a doctor for a blood test (HbA1c / fasting glucose).")
elif flag:
    st.error(f"Risk is above the screening threshold ({cfg['threshold']:.0%}). A blood test with a doctor is recommended.")
else:
    st.success(f"Risk is below the screening threshold ({cfg['threshold']:.0%}). Keep up regular check-ups.")

st.subheader("What drives this result")
st.caption("Each bar shows how much the risk changes if that one answer is replaced by the healthiest answer (or the training average for counts). Positive = raises your risk.")
healthy = {"BMI": 23, "MentHlth": 0, "PhysHlth": 0, "GenHlth": 1, "HighBP": 0, "HighChol": 0, "Smoker": 0, "Stroke": 0,
           "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "DiffWalk": 0,
           "NoDocbcCost": 0}
contrib = {}
for f, v in healthy.items():
    if row[f] != v:
        alt = dict(row, **{f: v})
        contrib[LABELS[f]] = risk - float(mc_risk(preprocess(alt, cfg), cfg, params)[0][0])
if contrib:
    s = pd.Series(contrib).sort_values(ascending=False)
    st.bar_chart(s.rename("risk change"), horizontal=True)
else:
    st.write("All modifiable answers are already at the healthiest option.")

st.subheader("What-if: change modifiable factors")
w1, w2, w3, w4 = st.columns(4)
new_bmi = w1.slider("BMI", 12, 60, int(min(row["BMI"], 60)))
new_pa = w2.toggle("Physically active", value=bool(row["PhysActivity"]))
new_bp = w3.toggle("High blood pressure", value=bool(row["HighBP"]))
new_ch = w4.toggle("High cholesterol", value=bool(row["HighChol"]))
alt = dict(row, BMI=new_bmi, PhysActivity=int(new_pa), HighBP=int(new_bp), HighChol=int(new_ch))
m2, s2 = mc_risk(preprocess(alt, cfg), cfg, params)
st.metric("Risk with these changes", f"{float(m2[0]):.1%}", delta=f"{(float(m2[0]) - risk) * 100:+.1f} percentage points", delta_color="inverse")

st.divider()
st.caption("Disclaimer: trained on self-reported 2015 US survey data; ROC-AUC about 0.82 on held-out data. It cannot diagnose diabetes, "
           "and results may be less reliable for groups under-represented in the survey. Always confirm with a clinical test.")
