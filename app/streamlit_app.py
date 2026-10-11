import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "students.csv"

TARGET = "final_score"
HOURS_COL = "hours_studied"
AT_RISK_BELOW = 50

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_models():
    model = joblib.load(MODELS / "best_model.joblib")
    risk = joblib.load(MODELS / "risk_model.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    return model, risk, meta


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


model, risk_model, meta = load_models()
df = load_data()

st.title("🎓 Student Performance Analytics & Early-Warning System")
st.caption("Exam score predict karo, at-risk students pehchano, aur what-if analysis karo.")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Student Details")
inputs = {}
for col, r in meta["numeric"].items():
    step = 1.0 if (r["max"] - r["min"]) > 20 else 0.1
    inputs[col] = st.sidebar.slider(
        col.replace("_", " ").title(), float(r["min"]), float(r["max"]), float(r["median"]), step=step
    )
for col, options in meta["categorical"].items():
    inputs[col] = st.sidebar.selectbox(col.replace("_", " ").title(), options)

row = pd.DataFrame([inputs])[meta["columns"]]
score = float(np.clip(model.predict(row)[0], 0, 100))
risk = float(risk_model.predict_proba(row)[0][1])

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "🔮 What-if", "📊 Class Insights", "🏆 Model Report"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    c1, c2 = st.columns(2)
    c1.metric("Predicted Final Score", f"{score:.1f} / 100")
    c2.metric("At-Risk Probability", f"{risk * 100:.1f}%")
    st.progress(min(int(score), 100))
    if risk >= 0.5:
        st.error("⚠️ Ye student At-Risk hai. Extra support/mentoring ki zarurat hai.")
    else:
        st.success("✅ Student safe zone mein hai.")

# ---------------- Tab 2: What-if ----------------
with tab2:
    if HOURS_COL in inputs:
        extra_hrs = st.slider("Agar extra ghante padhe (hours/day):", 0.0, 4.0, 2.0, 0.5)
        hmax = meta["numeric"][HOURS_COL]["max"]
        new_row = row.copy()
        new_row[HOURS_COL] = min(float(row[HOURS_COL].iloc[0]) + extra_hrs, hmax)
        new_score = float(np.clip(model.predict(new_row)[0], 0, 100))
        st.metric("New Predicted Score", f"{new_score:.1f}", delta=f"{new_score - score:+.1f}")

        hrs = np.linspace(meta["numeric"][HOURS_COL]["min"], hmax, 30)
        sweep = pd.concat([row] * len(hrs), ignore_index=True)
        sweep[HOURS_COL] = hrs
        preds = model.predict(sweep)
        fig = px.line(x=hrs, y=preds, labels={"x": "Hours Studied", "y": "Predicted Score"},
                      title="Study hours badhne pe score kaise badhta hai")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"'{HOURS_COL}' column nahi mila, app.py mein HOURS_COL badlo.")

# ---------------- Tab 3: Class insights ----------------
with tab3:
    at_risk_pct = (df[TARGET] < AT_RISK_BELOW).mean() * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Students", len(df))
    m2.metric("Average Score", f"{df[TARGET].mean():.1f}")
    m3.metric("At-Risk %", f"{at_risk_pct:.1f}%")

    st.plotly_chart(px.histogram(df, x=TARGET, nbins=30, title="Score Distribution"),
                    use_container_width=True)
    for cat in meta["categorical"]:
        g = df.groupby(cat)[TARGET].mean().reset_index()
        st.plotly_chart(px.bar(g, x=cat, y=TARGET, title=f"Average Score by {cat}"),
                        use_container_width=True)
    corr = df.select_dtypes("number").corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              title="Correlation Heatmap"), use_container_width=True)

# ---------------- Tab 4: Model report ----------------
with tab4:
    comp = REPORTS / "model_comparison.csv"
    if comp.exists():
        st.subheader("Model Comparison")
        st.dataframe(pd.read_csv(comp))
    for title, fname in [("SHAP - Feature Impact", "shap_summary.png"),
                         ("Actual vs Predicted", "actual_vs_predicted.png"),
                         ("Confusion Matrix (At-Risk)", "confusion_matrix.png")]:
        p = REPORTS / fname
        if p.exists():
            st.subheader(title)
            st.image(str(p))