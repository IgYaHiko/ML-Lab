"""
LAB 5: MULTIPLE LINEAR REGRESSION
Dataset: USA_Housing_Synthetic_10k.csv (10,000 synthetic records)

Run:
    pip install pandas numpy scikit-learn
    python lab5_usa_housing_mlr.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("USA_Housing_Synthetic_10k.csv")
df = df.dropna().copy()

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Price is the target variable. Address is excluded from the predictors.
target = "Price"
all_features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]
three_features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms"
]

X_all = df[all_features]
X_three = df[three_features]
y = df[target]

print("\nQuestion 1 features:", all_features)
print("Question 2 features:", three_features)


# 2. Split, normalize, train and evaluate
def evaluate_model(X, y, random_state=42, show_details=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    # StandardScaler is fitted on training data only.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    if show_details:
        scaler = model.named_steps["scaler"]
        reg = model.named_steps["regression"]

        # Convert coefficients back to original feature units.
        coefficients = reg.coef_ / scaler.scale_
        intercept = reg.intercept_ - np.sum(
            reg.coef_ * scaler.mean_ / scaler.scale_
        )

        print("\nRandom state:", random_state)
        print(f"Beta 0 (intercept): {intercept:.4f}")
        print("\nRegression coefficients:")
        for feature, beta in zip(X.columns, coefficients):
            print(f"Beta for {feature}: {beta:.4f}")

        print("\nEvaluation metrics:")
        print(f"MAE  : {mae:,.4f}")
        print(f"MSE  : {mse:,.4f}")
        print(f"RMSE : {rmse:,.4f}")
        print(f"R2   : {r2:.6f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# Question 1: MLR using all numerical features except Address
print("\n" + "=" * 65)
print("QUESTION 1: ALL FEATURES EXCEPT ADDRESS")
print("=" * 65)
q1_metrics = evaluate_model(X_all, y, random_state=42)

# Question 2: MLR using any three features except Address
print("\n" + "=" * 65)
print("QUESTION 2: THREE SELECTED FEATURES")
print("=" * 65)
q2_metrics = evaluate_model(X_three, y, random_state=42)


# Question 3: Test 200 random_state values and choose highest R2
def find_best_random_state(X, y, model_name, output_file):
    results = []

    for state in range(200):
        metrics = evaluate_model(
            X, y, random_state=state, show_details=False
        )
        results.append({"random_state": state, **metrics})

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["R2"].idxmax()]

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)
    print("Best random_state:", int(best["random_state"]))
    print(f"Best R2: {best['R2']:.6f}")
    print(f"MAE: {best['MAE']:,.4f}")
    print(f"MSE: {best['MSE']:,.4f}")
    print(f"RMSE: {best['RMSE']:,.4f}")

    results_df.to_csv(output_file, index=False)
    print("All 200 results saved to:", output_file)

    return int(best["random_state"]), results_df


best_state_all, results_all = find_best_random_state(
    X_all, y,
    "QUESTION 3: ALL FIVE FEATURES",
    "q1_all_features_200_splits.csv"
)

best_state_three, results_three = find_best_random_state(
    X_three, y,
    "QUESTION 3: THREE FEATURES",
    "q2_three_features_200_splits.csv"
)


# Question 4: Normalization
# StandardScaler normalization is applied in the Pipeline above.
print("\n" + "=" * 65)
print("QUESTION 4: FEATURE NORMALIZATION")
print("=" * 65)
print("StandardScaler formula: z = (x - mean) / standard deviation")
print("Scaler is fitted on training data only to prevent data leakage.")


# Final comparison
comparison = pd.DataFrame([
    {
        "Model": "All five features except Address",
        "Best random_state": best_state_all,
        "Best R2": results_all["R2"].max()
    },
    {
        "Model": "Three selected features",
        "Best random_state": best_state_three,
        "Best R2": results_three["R2"].max()
    }
])

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

if results_all["R2"].max() > results_three["R2"].max():
    print("\nConclusion: The all-feature model has the higher R2 score.")
elif results_all["R2"].max() < results_three["R2"].max():
    print("\nConclusion: The three-feature model has the higher R2 score.")
else:
    print("\nConclusion: Both models have the same best R2 score.")

print("\nLab 5 completed successfully.")
