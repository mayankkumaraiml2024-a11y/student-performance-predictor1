import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DATA_PATH, REPORTS_DIR, TARGET

out = REPORTS_DIR / "eda"
out.mkdir(exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(out / name, dpi=150)
    plt.close("all")


df = pd.read_csv(DATA_PATH)
print(df.shape)
print(df.describe().T)
print("\nMissing values:\n", df.isna().sum())

num = df.select_dtypes("number")

# 1. Target distribution
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], kde=True)
plt.title("Final Score Distribution")
save("01_score_distribution.png")

# 2. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
save("02_correlation_heatmap.png")

# 3. Boxplots (outliers)
rows = math.ceil(num.shape[1] / 3)
num.plot(kind="box", subplots=True, layout=(rows, 3), figsize=(12, 4 * rows))
save("03_boxplots.png")

# 4. Hours vs Score
if "hours_studied" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x="hours_studied", y=TARGET, scatter_kws={"alpha": 0.3})
    plt.title("Hours Studied vs Final Score")
    save("04_hours_vs_score.png")

# 5. Outliers count (IQR method)
print("\nOutliers (IQR method):")
for c in num.columns:
    q1, q3 = num[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)).sum()
    print(f"  {c}: {n_out}")

print("\nEDA plots saved in:", out)