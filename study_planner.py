# study planner project - BYOP submission
# basically the idea is: i always waste time studying the wrong subjects
# so i thought why not use ML to figure out what to focus on

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# generating data since i dont have a real dataset
# made 200 rows, each row = one subject a student might have

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "subject_difficulty": np.random.randint(1, 11, n),      # 1 to 10
    "days_until_exam": np.random.randint(1, 31, n),
    "past_score": np.random.randint(30, 101, n),            # out of 100
    "daily_available_hours": np.random.uniform(1, 8, n)
})

# this formula made the most sense to me
# if subject is hard + exam is close + i scored badly before = more hours needed
df["recommended_hours"] = (
    0.4 * df["subject_difficulty"]
    - 0.05 * df["days_until_exam"]
    - 0.03 * df["past_score"]
    + 0.2 * df["daily_available_hours"]
    + np.random.normal(0, 0.3, n)
).clip(0.5, 7.0).round(2)

print("sample data:")
print(df.head())


# LINEAR REGRESSION
# predicting how many hours to study based on the 4 features

X = df[["subject_difficulty", "days_until_exam", "past_score", "daily_available_hours"]]
y = df["recommended_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

preds = reg.predict(X_test)

print("\nRegression Results:")
print("MAE:", round(mean_absolute_error(y_test, preds), 2))
print("R2 score:", round(r2_score(y_test, preds), 2))

# coefficients - shows which feature matters most
print("\ncoefficients:")
for col, coef in zip(X.columns, reg.coef_):
    print(f"  {col} -> {coef:.4f}")


# KMEANS CLUSTERING
# grouping subjects into high / medium / low priority
# tried using all 4 features first but removing daily_available_hours gave cleaner clusters

# cols_for_cluster = ["subject_difficulty", "days_until_exam", "past_score", "daily_available_hours"]
cols_for_cluster = ["subject_difficulty", "days_until_exam", "past_score"]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[cols_for_cluster])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_data)

# labelling the clusters - whichever has highest avg study hours = high priority
means = df.groupby("cluster")["recommended_hours"].mean().sort_values(ascending=False)
labels = {
    means.index[0]: "High Priority",
    means.index[1]: "Medium Priority",
    means.index[2]: "Low Priority"
}
df["priority"] = df["cluster"].map(labels)

print("\ncluster breakdown:")
print(df.groupby("priority").agg(
    count=("subject_difficulty", "count"),
    avg_difficulty=("subject_difficulty", "mean"),
    avg_days_left=("days_until_exam", "mean"),
    avg_score=("past_score", "mean"),
    avg_hours=("recommended_hours", "mean")
).round(2).to_string())


# testing on one subject to see if it works
# using a subject i'm actually struggling with rn as an example

test = pd.DataFrame([{
    "subject_difficulty": 8,
    "days_until_exam": 5,
    "past_score": 55,
    "daily_available_hours": 4
}])

hrs = reg.predict(test)[0]
cluster_pred = kmeans.predict(scaler.transform(test[cols_for_cluster]))[0]
priority = labels[cluster_pred]

print(f"\nFor the test subject:")
print(f"  recommended study time : {hrs:.2f} hrs/day")
print(f"  priority group         : {priority}")

df.to_csv("study_planner_results.csv", index=False)
print("\ndone. saved results to csv")
