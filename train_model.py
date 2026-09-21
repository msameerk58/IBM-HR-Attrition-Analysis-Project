"""
train_model.py — Standalone model training script
Author : Mohammed Sameer Khazi
Course : AICTE IBM SkillsBuild Internship 2026

Run:  python train_model.py
Outputs: model.pkl  +  report_images/*.png  (6 charts)
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUTDIR  = os.path.join(BASE, "report_images")
MODEL   = os.path.join(BASE, "model.pkl")
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("[1/8] Loading dataset …")
df = pd.read_csv(DATA)
print(f"      Shape: {df.shape}")

# ── 2. Chart 1 — Attrition Distribution ───────────────────────────────────────
print("[2/8] Chart 1 — Attrition distribution …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df["Attrition"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Attrition Count", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Attrition"); axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Attrition Proportion", fontsize=14, fontweight="bold")
plt.suptitle("Employee Attrition Distribution", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart1_attrition_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Chart 2 — Age & Income ─────────────────────────────────────────────────
print("[3/8] Chart 2 — Age & income …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color in zip(["No", "Yes"], ["#3498db", "#e74c3c"]):
    axes[0].hist(df[df["Attrition"] == label]["Age"], bins=20,
                 alpha=0.6, label=label, color=color, edgecolor="white")
axes[0].set_title("Age Distribution by Attrition", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")
axes[0].legend(title="Attrition")
for label, color in zip(["No", "Yes"], ["#3498db", "#e74c3c"]):
    axes[1].hist(df[df["Attrition"] == label]["MonthlyIncome"], bins=20,
                 alpha=0.6, label=label, color=color, edgecolor="white")
axes[1].set_title("Monthly Income Distribution by Attrition", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Monthly Income"); axes[1].set_ylabel("Count")
axes[1].legend(title="Attrition")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart2_age_income_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Chart 3 — Department & Job Role ────────────────────────────────────────
print("[4/8] Chart 3 — Department / job role …")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dept = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack() * 100
dept.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="white")
axes[0].set_title("Attrition Rate by Department", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Department"); axes[0].set_ylabel("Percentage (%)")
axes[0].legend(title="Attrition"); axes[0].tick_params(axis="x", rotation=15)
role = df.groupby("JobRole")["Attrition"].value_counts(normalize=True).unstack() * 100
role["Yes"].sort_values().plot(kind="barh", ax=axes[1], color="#e74c3c", edgecolor="white")
axes[1].set_title("Attrition Rate by Job Role", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Attrition Rate (%)"); axes[1].set_ylabel("Job Role")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart3_dept_jobrole_attrition.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Chart 4 — Correlation heatmap ──────────────────────────────────────────
print("[5/8] Chart 4 — Correlation heatmap …")
df_corr = df.copy()
df_corr["Attrition_num"] = (df_corr["Attrition"] == "Yes").astype(int)
num_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
fig, ax = plt.subplots(figsize=(16, 12))
corr = df_corr[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart4_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 6. Chart 5 — WorkLife & OverTime ──────────────────────────────────────────
print("[6/8] Chart 5 — Work-life balance & overtime …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
wlb = df.groupby("WorkLifeBalance")["Attrition"].value_counts(normalize=True).unstack() * 100
wlb.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="white")
axes[0].set_title("Attrition by Work-Life Balance", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Work-Life Balance (1=Bad, 4=Best)"); axes[0].set_ylabel("Percentage (%)")
axes[0].legend(title="Attrition"); axes[0].tick_params(axis="x", rotation=0)
ot = df.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack() * 100
ot.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"], edgecolor="white")
axes[1].set_title("Attrition by OverTime", fontsize=13, fontweight="bold")
axes[1].set_xlabel("OverTime"); axes[1].set_ylabel("Percentage (%)")
axes[1].legend(title="Attrition"); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart5_worklife_overtime_attrition.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 7. Clean & train ──────────────────────────────────────────────────────────
print("[7/8] Cleaning, encoding, SMOTE, training …")
df.drop(columns=["EmployeeCount", "StandardHours", "Over18"], inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

model_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, class_weight="balanced",
    random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred  = model_rf.predict(X_test)
y_proba = model_rf.predict_proba(X_test)[:, 1]
acc  = accuracy_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_proba)

print(f"\n      Accuracy : {acc:.4f}")
print(f"      ROC-AUC  : {roc:.4f}")
print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))

# ── 8. Chart 6 — Confusion matrix & feature importance ────────────────────────
print("[8/8] Chart 6 — Confusion matrix + feature importance …")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Attrition", "Attrition"],
            yticklabels=["No Attrition", "Attrition"],
            linewidths=1, linecolor="white")
axes[0].set_title(f"Confusion Matrix\n(Accuracy: {acc:.2%})", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")
importances = pd.Series(model_rf.feature_importances_, index=X.columns)
importances.nlargest(15).sort_values().plot(
    kind="barh", ax=axes[1], color="#3498db", edgecolor="white")
axes[1].set_title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "chart6_confusion_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Save model ─────────────────────────────────────────────────────────────────
joblib.dump(model_rf, MODEL)
print(f"\n[OK] model.pkl saved  -> {MODEL}")
print(f"[OK] 6 charts saved   -> {OUTDIR}")
print("    " + "\n    ".join(sorted(os.listdir(OUTDIR))))
