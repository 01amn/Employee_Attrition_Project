#!/usr/bin/env python3
"""
Employee Attrition Analysis & Prediction — Green Destinations
-------------------------------------------------------------
Author: <Aman Mishra>
Use-case: Internship submission project (Data Science)
Dataset: Green Destinations

"""

import os
import sys
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------- CONFIG -------------------------
# Update this path if you want to run locally on Windows:
# For example: r"C:\Users\AMAN MISHRA\greendestination (1) (1).csv"
CSV_PATH = os.environ.get("GREEN_DESTINATIONS_CSV", r"C:\Users\AMAN MISHRA\greendestination (1) (1).csv")
# When running in this sandbox, we'll override CSV_PATH with the copied dataset if available.
SANDBOX_DEFAULT = r"/mnt/data/greendestination (1) (1).csv"
if os.path.exists(SANDBOX_DEFAULT):
    CSV_PATH = SANDBOX_DEFAULT

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- LOADING -------------------------
df = pd.read_csv(CSV_PATH)

# ------------------------- CLEANING -------------------------
# Drop clearly non-informative columns
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
for c in DROP_COLS:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# Ensure column names we need are present
required_cols = ["Attrition", "Age", "YearsAtCompany", "MonthlyIncome"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

# Binary encode Attrition ("Yes"->1, "No"->0)
df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})
if df["AttritionFlag"].isna().any():
    # Fallback: try lower/strip
    df["AttritionFlag"] = df["Attrition"].astype(str).str.strip().str.lower().map({"yes":1,"no":0})

# ------------------------- METRIC: ATTRITION RATE -------------------------
attrition_rate = df["AttritionFlag"].mean() * 100.0

# Save a small JSON of key numbers
key_numbers = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "attrition_rate_percent": round(float(attrition_rate), 2)
}
with open(OUTPUT_DIR / "key_numbers.json", "w") as f:
    json.dump(key_numbers, f, indent=2)

# ------------------------- EDA PLOTS (matplotlib-only) -------------------------

# 1) Overall Attrition Rate Bar
fig1 = plt.figure()
plt.bar(["Stayed", "Left"], [100-attrition_rate, attrition_rate])
plt.title("Overall Attrition Rate (%)")
plt.ylabel("Percent")
plt.savefig(OUTPUT_DIR / "1_attrition_rate_bar.png", bbox_inches="tight")
plt.close(fig1)

# 2) Age distribution split by Attrition
fig2 = plt.figure()
# Two histograms on the same axes (defaults color, no seaborn)
age_left = df.loc[df["AttritionFlag"]==1, "Age"]
age_stay = df.loc[df["AttritionFlag"]==0, "Age"]
plt.hist(age_stay, bins=15, alpha=0.6, label="Stayed")
plt.hist(age_left, bins=15, alpha=0.6, label="Left")
plt.title("Age Distribution by Attrition")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.savefig(OUTPUT_DIR / "2_age_by_attrition.png", bbox_inches="tight")
plt.close(fig2)

# 3) Monthly Income vs YearsAtCompany (two scatters)
fig3 = plt.figure()
stay = df[df["AttritionFlag"]==0]
left = df[df["AttritionFlag"]==1]
plt.scatter(stay["YearsAtCompany"], stay["MonthlyIncome"], alpha=0.5, label="Stayed")
plt.scatter(left["YearsAtCompany"], left["MonthlyIncome"], alpha=0.8, label="Left")
plt.title("Monthly Income vs YearsAtCompany")
plt.xlabel("YearsAtCompany")
plt.ylabel("MonthlyIncome")
plt.legend()
plt.savefig(OUTPUT_DIR / "3_income_vs_tenure.png", bbox_inches="tight")
plt.close(fig3)

# ------------------------- MODELING -------------------------
features = df[["Age", "YearsAtCompany", "MonthlyIncome"]].copy()
target = df["AttritionFlag"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
log_reg.fit(X_train_s, y_train)

y_pred = log_reg.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

# Save metrics to text
with open(OUTPUT_DIR / "model_metrics.txt", "w") as f:
    f.write("Logistic Regression — Features: Age, YearsAtCompany, MonthlyIncome\n")
    f.write(f"Accuracy: {acc:.3f}\n\n")
    f.write("Confusion Matrix [ [TN FP]\n                   [FN TP] ]\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# ------------------------- FEATURE EFFECTS (simple) -------------------------
coefs = pd.Series(log_reg.coef_[0], index=features.columns)
coefs_df = pd.DataFrame({"Feature": coefs.index, "Coefficient": coefs.values})
coefs_df.to_csv(OUTPUT_DIR / "coefficients.csv", index=False)

# Save a coefficient bar chart (no custom colors)
fig4 = plt.figure()
plt.bar(coefs.index, coefs.values)
plt.title("Logistic Regression Coefficients")
plt.ylabel("Weight (standardized)")
plt.savefig(OUTPUT_DIR / "4_logreg_coefficients.png", bbox_inches="tight")
plt.close(fig4)

# ------------------------- REPORT -------------------------
summary = f"""
# Green Destinations — Attrition Analysis Summary

- **Rows**: {df.shape[0]}
- **Columns**: {df.shape[1]}
- **Attrition Rate**: {attrition_rate:.2f}%

## Model (Logistic Regression)
- Features used: Age, YearsAtCompany, MonthlyIncome
- Accuracy: {acc:.3f}

### Quick Interpretation
- Positive coefficient => higher value increases odds of leaving
- Negative coefficient => higher value decreases odds of leaving

Coefficients (standardized):
{coefs.round(3).to_string()}

Check `outputs/` for figures and detailed metrics.
"""
with open(OUTPUT_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
    f.write(summary.strip())

print("Project run complete.")
print(f"Attrition Rate: {attrition_rate:.2f}%")
print(f"Accuracy: {acc:.3f}")
print("Outputs saved in:", OUTPUT_DIR.resolve())
