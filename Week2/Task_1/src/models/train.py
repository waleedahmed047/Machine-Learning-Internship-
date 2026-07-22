"""
Week 02 Internship Project
Regression Model Training
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.dummy import DummyRegressor

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from src.features.build_features import FeatureBuilder


print("=" * 70)
print("LINEAR MODEL TRAINING")
print("=" * 70)

##########################################################
# Create Folders
##########################################################

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

##########################################################
# Load Dataset
##########################################################

print("\nLoading cleaned dataset...")

df = pd.read_csv(
    "data/processed/clean_data.csv"
)

print(f"Original Dataset Shape : {df.shape}")

##########################################################
# Fast Training Sample
##########################################################

# Large dataset ko sample karenge taake training fast ho
if len(df) > 50000:

    df = df.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

print(f"Training Dataset Shape : {df.shape}")

##########################################################
# Target
##########################################################

TARGET = "trip_duration"

X = df.drop(columns=[TARGET])

y = df[TARGET]

##########################################################
# Feature Engineering
##########################################################

print("\nBuilding Features...")

builder = FeatureBuilder()

X = builder.fit_transform(X)

print("Feature Engineering Completed.")

##########################################################
# Feature Lists
##########################################################

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print(f"\nNumeric Features     : {len(numeric_features)}")
print(f"Categorical Features : {len(categorical_features)}")

##########################################################
# Numeric Pipeline
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

##########################################################
# Categorical Pipeline
##########################################################

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)

##########################################################
# Combined Preprocessor
##########################################################

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

print("\nTrain Shape :", X_train.shape)

print("Test Shape  :", X_test.shape)

##########################################################
# Models
##########################################################

models = {

    "Dummy Regressor": DummyRegressor(),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0,
        random_state=42
    ),

    "Lasso Regression": Lasso(
        alpha=0.01,
        max_iter=2000,
        random_state=42
    ),

    "Elastic Net": ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=2000,
        random_state=42
    )

}

##########################################################
# Results
##########################################################

results = []

best_model = None

best_score = float("-inf")

##########################################################
# Train Models
##########################################################

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training : {model_name}")
    print("=" * 70)

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )

    ######################################################
    # Train
    ######################################################

    pipeline.fit(
        X_train,
        y_train
    )

    ######################################################
    # Prediction
    ######################################################

    prediction = pipeline.predict(
        X_test
    )

    ######################################################
    # Metrics
    ######################################################

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

    ######################################################
    # Save Results
    ######################################################

    results.append({

        "Model": model_name,

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "R2 Score": round(r2, 4)

    })

    ######################################################
    # Print Results
    ######################################################

    print(f"MAE      : {mae:.4f}")

    print(f"RMSE     : {rmse:.4f}")

    print(f"R2 Score : {r2:.4f}")

    ######################################################
    # Save Best Model
    ######################################################

    if r2 > best_score:

        best_score = r2

        best_model = pipeline

        print("Current Best Model Updated.")
        ##########################################################
# Model Comparison
##########################################################

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

##########################################################
# Save Model Comparison
##########################################################

comparison_path = os.path.join(
    "reports",
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

##########################################################
# Save Champion Model
##########################################################

model_path = os.path.join(
    "models",
    "champion_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)

##########################################################
# Final Summary
##########################################################

best_model_name = results_df.iloc[0]["Model"]
best_r2 = results_df.iloc[0]["R2 Score"]

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Best Model : {best_model_name}")
print(f"Best R²    : {best_r2:.4f}")

print("\nFiles Generated:")

print(f"✓ {comparison_path}")
print(f"✓ {model_path}")

print("\nTop Models Ranking:")

for index, row in results_df.iterrows():

    print(
        f"{index + 1}. "
        f"{row['Model']} | "
        f"R² = {row['R2 Score']:.4f}"
    )

print("\nLinear model training completed successfully.")

print("=" * 70)