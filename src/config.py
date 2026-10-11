from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "students.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "final_score"      # real dataset lo to yahan target column ka naam badlo
AT_RISK_BELOW = 50          # is score se kam = At-Risk student
RANDOM_STATE = 42

for d in (DATA_PATH.parent, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)