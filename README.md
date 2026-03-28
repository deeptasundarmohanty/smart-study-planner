# Smart Study Session Planner

A machine learning project that helps students allocate study time more effectively using **Linear Regression** (supervised learning) and **KMeans Clustering** (unsupervised learning).

---

## The Problem

Students often struggle to decide *how much* time to spend studying each subject. Without a data-driven approach, they tend to over-prepare for subjects they already know well, and under-prepare for harder or closer exams. This project builds a simple ML system that recommends study hours per subject and groups subjects by priority — so you always know what to focus on first.

---

## What It Does

| Feature | Technique Used |
|---|---|
| Predict recommended study hours per subject | Linear Regression (Supervised Learning) |
| Group subjects into High / Medium / Low priority | KMeans Clustering (Unsupervised Learning) |

**Inputs per subject:**
- `subject_difficulty` — rated 1 (easy) to 10 (hard)
- `days_until_exam` — how many days remain
- `past_score` — your previous score in that subject (%)
- `daily_available_hours` — free hours available per day

**Outputs:**
- Recommended study hours per day for that subject
- Priority group: High, Medium, or Low

---

## Project Structure

```
study_planner/
│
├── study_planner.py          # Main script (data gen, model training, prediction)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/deeptasundarmohanty/smart-study-planner.git
cd smart-study-planner
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python study_planner.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
```

Or simply run:
```bash
pip install pandas numpy scikit-learn
```

---

## Sample Output

```
=======================================================
   Smart Study Session Planner
=======================================================

[Dataset] 200 synthetic student-subject records generated.

PART A: Supervised Learning – Predicting Study Hours
  MAE  : 0.3331 hours
  R2   : 0.6480

PART B: Unsupervised Learning – Subject Priority Clusters
                 count  avg_difficulty  avg_days_left  avg_recommended_hrs
High Priority       80            6.84           6.50                 1.38
Medium Priority     62            5.52          22.15                 1.21
Low Priority        58            3.84          21.00                 0.54

PART C: Predict for a New Subject (Example)
  subject_difficulty           : 8
  days_until_exam              : 5
  past_score                   : 55
  daily_available_hours        : 4

  Predicted Recommended Study Hours : 1.90 hrs/day
  Predicted Priority Group          : High Priority
```

---

## How to Customise for Your Own Data

To predict for your own subjects, edit the `new_subject` dictionary in `study_planner.py`:

```python
new_subject = {
    "subject_difficulty":    7,   # Change this
    "days_until_exam":       10,  # Change this
    "past_score":            65,  # Change this
    "daily_available_hours": 3    # Change this
}
```

Then re-run the script.

---

## Concepts Applied

- **Linear Regression** — learns the relationship between subject features and study time
- **KMeans Clustering** — groups subjects into 3 priority tiers without labeled data
- **StandardScaler** — normalizes features before clustering for fair distance calculation
- **Train/Test Split** — evaluates model on unseen data to prevent overfitting
- **MAE & R² Score** — metrics to measure prediction quality

---

## Course

**Fundamentals of AI and ML** — BYOP (Bring Your Own Project) Capstone Submission

---

## Author

Your Name — VIT Student
