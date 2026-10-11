import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from utils import build_preprocessor, split_columns

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]
num_cols, cat_cols = split_columns(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                     random_state=RANDOM_STATE)
except ImportError:
    print("xgboost installed nahi hai, skip kar rahe hain")

rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("model", model)])
    pipe.fit(X_train, y_train)
    metrics = evaluate(y_test, pipe.predict(X_test))
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2").mean()
    rows.append({"Model": name, **metrics, "CV_R2": cv_r2})
    print(f"{name:20s} MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.3f}")

# ---------- Hyperparameter tuning (Gradient Boosting) ----------
print("\nGridSearchCV chal raha hai...")
gb_pipe = Pipeline([
    ("prep", build_preprocessor(num_cols, cat_cols)),
    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
}
search = GridSearchCV(gb_pipe, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)

best_metrics = evaluate(y_test, best.predict(X_test))
best_cv = cross_val_score(best, X_train, y_train, cv=kf, scoring="r2").mean()
rows.append({"Model": "Gradient Boosting (Tuned)", **best_metrics, "CV_R2": best_cv})

results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
results.round(3).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
print("\n", results.round(3))

# ---------- Plots ----------
pred = best.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.4)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Score"); plt.ylabel("Predicted Score"); plt.title("Actual vs Predicted")
plt.tight_layout(); plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
plt.hist(y_test - pred, bins=30)
plt.xlabel("Residual (Actual - Predicted)"); plt.title("Residual Distribution")
plt.tight_layout(); plt.savefig(REPORTS_DIR / "residuals.png", dpi=150); plt.close()

# ---------- Save model + metadata for Streamlit ----------
joblib.dump(best, MODELS_DIR / "best_model.joblib")

meta = {
    "columns": X.columns.tolist(),
    "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                    "median": float(X[c].median())} for c in num_cols},
    "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
}
(MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
print("\nModel aur meta.json save ho gaye:", MODELS_DIR)