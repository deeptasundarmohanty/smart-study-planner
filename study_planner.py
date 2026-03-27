"""
Smart Study Session Planner
============================
Uses Supervised Learning (Linear Regression) to predict recommended study hours
and Unsupervised Learning (KMeans Clustering) to group subjects by priority.

Course: Fundamentals of AI and ML
Project: BYOP – Smart Study Session Planner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Generate Synthetic Dataset
# ─────────────────────────────────────────────

np.random.seed(42)
n = 200

data = pd.DataFrame({
    "subject_difficulty":    np.random.randint(1, 11, n),   # 1 (easy) to 10 (hard)
    "days_until_exam":       np.random.randint(1, 31, n),   # 1 to 30 days
    "past_score":            np.random.randint(30, 101, n), # previous exam score (%)
    "daily_available_hours": np.random.uniform(1, 8, n),    # hours free per day
})

# Target: recommended study hours per day (realistic formula + noise)
data["recommended_hours"] = (
    0.4 * data["subject_difficulty"]
    - 0.05 * data["days_until_exam"]
    - 0.03 * data["past_score"]
    + 0.2 * data["daily_available_hours"]
    + np.random.normal(0, 0.3, n)
).clip(0.5, 7.0).round(2)

print("=" * 55)
print("   Smart Study Session Planner")
print("=" * 55)
print(f"\n[Dataset] {n} synthetic student-subject records generated.")
print(data.head())

# ─────────────────────────────────────────────
# STEP 2: Supervised Learning – Linear Regression
# ─────────────────────────────────────────────

print("\n" + "-" * 55)
print("PART A: Supervised Learning – Predicting Study Hours")
print("-" * 55)

features = ["subject_difficulty", "days_until_exam", "past_score", "daily_available_hours"]
X = data[features]
y = data["recommended_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nModel Trained on {len(X_train)} samples, tested on {len(X_test)} samples.")
print(f"  Mean Absolute Error (MAE) : {mae:.4f} hours")
print(f"  R2 Score                  : {r2:.4f}")
print("\nCoefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<28} : {coef:+.4f}")
print(f"  {'Intercept':<28} : {model.intercept_:+.4f}")
