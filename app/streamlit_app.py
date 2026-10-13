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

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_models():
    model = joblib.load(MODELS / "best_model.joblib")
    risk_model = joblib.load(MODELS / "risk_model.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    return model, risk_model, meta


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


model, risk_model, meta = load_models()
df = load_data()

st.title("🎓 Student Performance Analytics & Early-Warning System")
st.caption(
    "Exam score predict karo, at-risk students pehchano, "
    "aur what-if analysis karo."
)

# ---------------- Sidebar: Student Details ----------------
st.sidebar.header("Student Details")

inputs = {}

for col, r in meta["numeric"].items():
    step = 1.0 if (r["max"] - r["min"]) > 20 else 0.1

    inputs[col] = st.sidebar.slider(
        col.replace("_", " ").title(),
        float(r["min"]),
        float(r["max"]),
        float(r["median"]),
        step=step,
    )

for col, options in meta["categorical"].items():
    inputs[col] = st.sidebar.selectbox(
        col.replace("_", " ").title(),
        options,
    )

row = pd.DataFrame([inputs])[meta["columns"]]

score = float(np.clip(model.predict(row)[0], 0, 100))
risk = float(risk_model.predict_proba(row)[0][1])

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Prediction",
        "🔮 What-if",
        "📊 Class Insights",
        "🏆 Model Report",
    ]
)

# ---------------- Tab 1: Prediction ----------------
with tab1:
    c1, c2 = st.columns(2)

    c1.metric("Predicted Final Score", f"{score:.1f} / 100")
    c2.metric("At-Risk Probability", f"{risk * 100:.1f}%")

    st.progress(min(max(int(score), 0), 100))

    if risk >= 0.5:
        st.error(
            "⚠️ Ye student At-Risk hai. "
            "Extra support/mentoring ki zarurat hai."
        )
    else:
        st.success("✅ Student safe zone mein hai.")

# ---------------- Tab 2: What-if Analysis ----------------
with tab2:
    if HOURS_COL in inputs and HOURS_COL in meta["numeric"]:
        extra_hrs = st.slider(
            "Agar extra ghante padhe (hours/day):",
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
        )

        hmin = float(meta["numeric"][HOURS_COL]["min"])
        hmax = float(meta["numeric"][HOURS_COL]["max"])

        current_hours = float(row[HOURS_COL].iloc[0])

        new_row = row.copy()
        new_row[HOURS_COL] = min(
            current_hours + extra_hrs,
            hmax,
        )

        new_score = float(
            np.clip(model.predict(new_row)[0], 0, 100)
        )

        st.metric(
            "New Predicted Score",
            f"{new_score:.1f}",
            delta=f"{new_score - score:+.1f}",
        )

        # Study hours ki values pehle generate karo
        hrs = np.linspace(hmin, hmax, 30)

        # Har study-hours value ke liye ek row
        sweep = pd.concat(
            [row] * len(hrs),
            ignore_index=True,
        )

        sweep[HOURS_COL] = hrs

        preds = np.clip(model.predict(sweep), 0, 100)

        fig = px.line(
            x=hrs,
            y=preds,
            labels={
                "x": "Hours Studied",
                "y": "Predicted Score",
            },
            title="Study Hours vs Predicted Score",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            f"'{HOURS_COL}' feature nahi mila. "
            "Model metadata aur feature names check karo."
        )

# ---------------- Tab 3: Class Insights ----------------
with tab3:
    if TARGET not in df.columns:
        st.error(f"Dataset mein '{TARGET}' column nahi mila.")
    else:
        at_risk_pct = (
            (df[TARGET] < AT_RISK_BELOW).mean() * 100
        )

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Students", len(df))
        m2.metric("Average Score", f"{df[TARGET].mean():.1f}")
        m3.metric("At-Risk %", f"{at_risk_pct:.1f}%")

        st.plotly_chart(
            px.histogram(
                df,
                x=TARGET,
                nbins=30,
                title="Score Distribution",
            ),
            use_container_width=True,
        )

        for cat in meta["categorical"]:
            if cat in df.columns:
                g = df.groupby(cat)[TARGET].mean().reset_index()

                st.plotly_chart(
                    px.bar(
                        g,
                        x=cat,
                        y=TARGET,
                        title=f"Average Score by {cat}",
                    ),
                    use_container_width=True,
                )

        corr = df.select_dtypes(include="number").corr()

        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
            ),
            use_container_width=True,
        )

# ---------------- Tab 4: Model Report ----------------
with tab4:
    comp = REPORTS / "model_comparison.csv"

    if comp.exists():
        st.subheader("Model Comparison")
        st.dataframe(pd.read_csv(comp), use_container_width=True)
    else:
        st.info("Model comparison report abhi available nahi hai.")

    reports = [
        ("SHAP - Feature Impact", "shap_summary.png"),
        ("Actual vs Predicted", "actual_vs_predicted.png"),
        ("Confusion Matrix (At-Risk)", "confusion_matrix.png"),
    ]

    for title, fname in reports:
        report_path = REPORTS / fname

        if report_path.exists():
            st.subheader(title)
            st.image(str(report_path), use_container_width=True)
