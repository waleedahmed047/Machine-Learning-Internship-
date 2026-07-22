"""
Week 02 Internship Project
Hyperparameter Tuning
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import FeatureBuilder


print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

##########################################################
# Create Folder
##########################################################

os.makedirs("models", exist_ok=True)

##########################################################
# Load Dataset
##########################################################

print("\nLoading Dataset...")

df = pd.read_csv(
    "data/processed/clean_data.csv"
)

print(f"Original Shape : {df.shape}")

##########################################################
# Fast Sample
##########################################################

if len(df) > 50000:

    df = df.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

print(f"Training Shape : {df.shape}")

##########################################################
# Features
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
# Feature Lists
##########################################################

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

##########################################################
# Pipelines
##########################################################

numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]

)

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ]

)

##########################################################
# Train Test Split
##########################################################

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print(f"\nTrain Shape : {X_train.shape}")
print(f"Test Shape  : {X_test.shape}")
##########################################################
# Random Forest Pipeline
##########################################################

pipeline = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "model",

            RandomForestRegressor(

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)

##########################################################
# Hyperparameter Grid
##########################################################

param_grid = {

    "model__n_estimators": [

        20,
        30,
        50,
        75

    ],

    "model__max_depth": [

        10,
        15,
        20,
        None

    ],

    "model__min_samples_split": [

        2,
        5,
        10

    ],

    "model__min_samples_leaf": [

        1,
        2,
        4

    ],

    "model__max_features": [

        "sqrt",
        "log2",
        None

    ]

}

##########################################################
# Randomized Search
##########################################################

print("\nStarting Hyperparameter Tuning...")

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=5,

    cv=3,

    scoring="r2",

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True

)

##########################################################
# Fit Search
##########################################################

search.fit(

    X_train,

    y_train

)

##########################################################
# Best Model
##########################################################

best_model = search.best_estimator_

print("\nHyperparameter Tuning Completed.")

print(f"\nBest Cross Validation Score : {search.best_score_:.4f}")

print("\nBest Parameters:\n")

for key, value in search.best_params_.items():

    print(f"{key} : {value}")
    from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

##########################################################
# Prediction
##########################################################

print("\nEvaluating Tuned Model...")

prediction = best_model.predict(
    X_test
)

##########################################################
# Metrics
##########################################################

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

r2 = r2_score(
    y_test,
    prediction
)

print("\n" + "=" * 70)
print("TUNED MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")

##########################################################
# Save Tuned Model
##########################################################

model_path = os.path.join(
    "models",
    "tuned_random_forest.joblib"
)

joblib.dump(
    best_model,
    model_path
)

##########################################################
# Save Best Parameters
##########################################################

best_params = pd.DataFrame({

    "Parameter": list(search.best_params_.keys()),

    "Value": list(search.best_params_.values())

})

report_path = os.path.join(
    "reports",
    "best_parameters.csv"
)

best_params.to_csv(
    report_path,
    index=False
)

##########################################################
# Final Summary
##########################################################

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)

print(f"Best CV Score : {search.best_score_:.4f}")
print(f"Test R² Score : {r2:.4f}")

print("\nGenerated Files:")

print(f"✓ {model_path}")
print(f"✓ {report_path}")

print("\nHyperparameter tuning completed successfully.")

print("=" * 70)