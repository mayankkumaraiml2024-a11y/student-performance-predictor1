# 🎓 Student Performance Analytics & Early-Warning System

Predicts exam scores and flags at-risk students using ML, with explainability (SHAP) and a Streamlit dashboard.

## Features
- Multi-feature regression (study hours, attendance, sleep, previous score, etc.)
- Preprocessing with ColumnTransformer + Pipeline (no data leakage)
- 6 models compared (MAE, RMSE, R²) + GridSearchCV + 5-fold CV
- SHAP feature importance
- At-Risk classification (Precision / Recall / F1 / Confusion Matrix)
- Streamlit app with what-if analysis

## Run locally
pip install -r requirements-dev.txt
python src/generate_data.py
python src/train.py
python src/train_classifier.py
python src/explain.py
streamlit run app/streamlit_app.py

## Live Demo
<yahan Streamlit link paste karna>

## Screenshots
<app aur SHAP plot ke screenshots>