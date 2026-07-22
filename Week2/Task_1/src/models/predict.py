"""
Week 03 Advanced Regression
Prediction Module
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

##########################################################
# Create Reports Folder
##########################################################

os.makedirs(
    "reports",
    exist_ok=True
)

##########################################################
# Load Champion Model
##########################################################

model = joblib.load(
    "models/champion_model.joblib"
)

##########################################################
# Load Dataset
##########################################################

df = pd.read_csv(
    "data/processed/clean_data.csv"
)

##########################################################
# Remove Target Column
##########################################################

TARGET = "trip_duration"

if TARGET in df.columns:

    X = df.drop(columns=[TARGET])

else:

    X = df.copy()

##########################################################
# Predict
##########################################################

prediction = model.predict(X)

##########################################################
# Save Predictions
##########################################################

result = X.copy()

result["Predicted_Trip_Duration"] = prediction

result.to_csv(
    "reports/predictions.csv",
    index=False
)

##########################################################
# Preview
##########################################################

print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print("\nFirst 10 Predictions:\n")

print(result.head(10))

print("\nPrediction file saved successfully.")

print("Location : reports/predictions.csv")