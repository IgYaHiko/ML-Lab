"""
LAB: LOGISTIC REGRESSION FOR CANCER CLASSIFICATION
Dataset: data.csv (10,000 synthetic records)

Install dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn

Run:
    python logistic_regression_cancer.py

IMPORTANT:
This is a synthetic educational dataset. It is NOT real patient data and
must NOT be used for medical diagnosis or clinical decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("data.csv")

print("=" * 70)
print("LOGISTIC REGRESSION: CANCER CLASSIFICATION")
print("=" * 70)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum().sum())

print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts())

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# Remove duplicate records.
df = df.drop_duplicates()

# Remove rows where the target label is missing.
df = df.dropna(subset=["diagnosis"]).copy()

# Convert diagnosis to binary labels:
# B = 0 (Benign), M = 1 (Malignant)
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Remove the unique sample ID because it is not a useful predictor.
X = df.drop(columns=["diagnosis", "sample_id"])
y = df["diagnosis"]

# Identify numeric and categorical input columns.
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric feature count:", len(numeric_features))
print("Categorical features:", categorical_features)

# ---------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ---------------------------------------------------------
# Correlation is calculated on numeric features and encoded target.
correlation_data = X[numeric_features].copy()
correlation_data["diagnosis"] = y

correlation_with_target = (
    correlation_data.corr(numeric_only=True)["diagnosis"]
    .drop("diagnosis")
    .sort_values(key=abs, ascending=False)
)

print("\nTop 10 features correlated with diagnosis:")
print(correlation_with_target.head(10))

# Draw a heatmap using the 10 most correlated features.
top_features = correlation_with_target.head(10).index.tolist()
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_data[top_features + ["diagnosis"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap: Top 10 Features and Diagnosis")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------------
# 80% data for training and 20% for testing.
# Stratification preserves the class distribution in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records :", X_test.shape[0])

# ---------------------------------------------------------
# 5. MISSING VALUE HANDLING, ENCODING AND NORMALIZATION
# ---------------------------------------------------------
# Numeric columns:
#   - Fill missing values using the training median.
#   - Normalize values using StandardScaler.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical columns:
#   - Fill missing values using the most frequent category.
#   - Convert categories into numeric one-hot encoded columns.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the correct preprocessing to each column type.
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ---------------------------------------------------------
# 6. BUILD AND TRAIN LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

# Predict class labels and probabilities.
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 7. CLASSIFICATION METRICS
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nDetailed classification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign (B)", "Malignant (M)"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 8. CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 9. ROC CURVE AND AUC SCORE
# ---------------------------------------------------------
false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test, y_probability
)
auc_score = roc_auc_score(y_test, y_probability)

print(f"\nROC-AUC Score: {auc_score:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"Logistic Regression (AUC = {auc_score:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 10. FINAL SUMMARY
# ---------------------------------------------------------
summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [accuracy, precision, recall, f1, auc_score]
})

print("\nFINAL SUMMARY:")
print(summary.to_string(index=False))

summary.to_csv("logistic_regression_metrics.csv", index=False)

print("\nGenerated files:")
print("1. correlation_heatmap.png")
print("2. confusion_matrix.png")
print("3. roc_curve.png")
print("4. logistic_regression_metrics.csv")

print("\nProgram completed successfully.")
