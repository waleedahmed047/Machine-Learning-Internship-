"""
Week 02 Internship Project
Model Evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.features.build_features import FeatureBuilder

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

##########################################################
# Create Report Folder
##########################################################

os.makedirs(
    "reports/figures",
    exist_ok=True
)

##########################################################
# Load Dataset
##########################################################

print("\nLoading Dataset...")

df = pd.read_csv(
    "data/processed/clean_data.csv"
)

print(f"Original Shape : {df.shape}")

##########################################################
# Fast Evaluation Sample
##########################################################

if len(df) > 50000:

    df = df.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

print(f"Evaluation Shape : {df.shape}")

##########################################################
# Target
##########################################################

TARGET = "trip_duration"

X = df.drop(columns=[TARGET])

y = df[TARGET]

##########################################################
# Feature Engineering
##########################################################

builder = FeatureBuilder()

X = builder.fit_transform(X)

##########################################################
# Load Champion Model
##########################################################

print("\nLoading Champion Model...")

model = joblib.load(
    "models/champion_model.joblib"
)

##########################################################
# Prediction
##########################################################

prediction = model.predict(X)

##########################################################
# Metrics
##########################################################

mae = mean_absolute_error(
    y,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        prediction
    )
)

r2 = r2_score(
    y,
    prediction
)

print("\nEvaluation Results")

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
##########################################################
# Actual vs Predicted Plot
##########################################################

print("\nGenerating Evaluation Graphs...")

plt.figure(figsize=(8, 6))

plt.scatter(

    y,

    prediction,

    alpha=0.5

)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "reports/figures/actual_vs_predicted.png",

    dpi=300

)

plt.close()

##########################################################
# Residual Plot
##########################################################

residuals = y - prediction

plt.figure(figsize=(8, 6))

plt.scatter(

    prediction,

    residuals,

    alpha=0.5

)

plt.axhline(

    y=0,

    linestyle="--"

)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "reports/figures/residual_plot.png",

    dpi=300

)

plt.close()

##########################################################
# Error Distribution
##########################################################

plt.figure(figsize=(8, 6))

sns.histplot(

    residuals,

    bins=40,

    kde=True

)

plt.title("Residual Distribution")

plt.xlabel("Residual")

plt.tight_layout()

plt.savefig(

    "reports/figures/error_distribution.png",

    dpi=300

)

plt.close()

##########################################################
# Prediction Distribution
##########################################################

plt.figure(figsize=(8, 6))

sns.histplot(

    prediction,

    bins=40,

    kde=True

)

plt.title("Prediction Distribution")

plt.xlabel("Predicted Trip Duration")

plt.tight_layout()

plt.savefig(

    "reports/figures/prediction_distribution.png",

    dpi=300

)

plt.close()

print("\nGraphs Generated Successfully.")
##########################################################
# Save Evaluation Metrics
##########################################################

metrics_df = pd.DataFrame({

    "Metric": [

        "MAE",
        "RMSE",
        "R2 Score"

    ],

    "Value": [

        round(mae, 4),
        round(rmse, 4),
        round(r2, 4)

    ]

})

metrics_path = os.path.join(

    "reports",

    "evaluation_metrics.csv"

)

metrics_df.to_csv(

    metrics_path,

    index=False

)

##########################################################
# Display Metrics
##########################################################

print("\n" + "=" * 70)
print("MODEL EVALUATION SUMMARY")
print("=" * 70)

print(metrics_df)

##########################################################
# Generated Files
##########################################################

print("\nGenerated Files:\n")

files = [

    "reports/evaluation_metrics.csv",

    "reports/figures/actual_vs_predicted.png",

    "reports/figures/residual_plot.png",

    "reports/figures/error_distribution.png",

    "reports/figures/prediction_distribution.png"

]

for file in files:

    print(f"✓ {file}")

##########################################################
# Completion
##########################################################

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Final MAE      : {mae:.4f}")
print(f"Final RMSE     : {rmse:.4f}")
print(f"Final R² Score : {r2:.4f}")

print("\nEvaluation reports and graphs have been saved.")

print("=" * 70)