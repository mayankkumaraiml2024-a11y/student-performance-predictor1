import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import AT_RISK_BELOW, DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from utils import build_preprocessor, split_columns

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = (df[TARGET] < AT_RISK_BELOW).astype(int)     # 1 = At-Risk
print("At-risk students: %.1f%%" % (y.mean() * 100))

num_cols, cat_cols = split_columns(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
}

best_name, best_pipe, best_f1 = None, None, -1
report_text = ""
for name, clf in candidates.items():
    pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    f1 = f1_score(y_test, pred)
    rep = classification_report(y_test, pred, target_names=["Safe", "At-Risk"])
    print(f"\n=== {name} (F1 At-Risk = {f1:.3f}) ===\n{rep}")
    report_text += f"=== {name} ===\n{rep}\n"
    if f1 > best_f1:
        best_name, best_pipe, best_f1 = name, pipe, f1

print("Best classifier:", best_name)
(REPORTS_DIR / "classification_report.txt").write_text(report_text)

ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test, display_labels=["Safe", "At-Risk"])
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150); plt.close()

joblib.dump(best_pipe, MODELS_DIR / "risk_model.joblib")
print("risk_model.joblib saved")