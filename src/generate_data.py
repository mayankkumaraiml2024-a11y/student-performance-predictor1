import numpy as np
import pandas as pd
from config import DATA_PATH, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)
N = 1500

hours = np.clip(rng.normal(6, 2.5, N), 0.5, 14).round(1)
attendance = np.clip(rng.normal(80, 12, N), 40, 100).round(0)
sleep = np.clip(rng.normal(6.8, 1.2, N), 3, 10).round(1)
prev = np.clip(rng.normal(65, 15, N), 20, 100).round(0)
assign = np.clip(rng.normal(75, 18, N), 10, 100).round(0)
internet = rng.choice(["Yes", "No"], N, p=[0.8, 0.2])
extra = rng.choice(["Yes", "No"], N, p=[0.4, 0.6])
parent_edu = rng.choice(["High School", "Graduate", "Postgraduate"], N, p=[0.4, 0.4, 0.2])

score = (
    -5
    + 11 * np.log1p(hours)              # padhai ka diminishing return
    + 0.2 * attendance
    - 1.0 * (sleep - 7.5) ** 2          # bahut kam/zyada neend = nuksaan
    + 0.3 * prev
    + 0.1 * assign
    + np.where(internet == "Yes", 2.0, 0)
    + np.where(extra == "Yes", 2.5, 0)
    + np.select([parent_edu == "Graduate", parent_edu == "Postgraduate"], [1.5, 3.0], 0)
    + rng.normal(0, 4, N)
)
score = np.clip(score, 0, 100).round(1)

df = pd.DataFrame({
    "hours_studied": hours,
    "attendance": attendance,
    "sleep_hours": sleep,
    "previous_score": prev,
    "assignments_completed": assign,
    "internet_access": internet,
    "extra_classes": extra,
    "parental_education": parent_edu,
    "final_score": score,
})

# thode missing values daalte hain taaki preprocessing (imputer) ka kaam dikhe
df.loc[df.sample(40, random_state=1).index, "sleep_hours"] = np.nan
df.loc[df.sample(30, random_state=2).index, "parental_education"] = np.nan

df.to_csv(DATA_PATH, index=False)
print("Dataset saved:", DATA_PATH, df.shape)