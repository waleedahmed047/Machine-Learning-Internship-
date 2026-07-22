"""
Week 02 Internship Project
Tree-Based Regression Models
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

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from src.features.build_features import FeatureBuilder

print("=" * 70)
print("TREE BASED MODEL TRAINING")
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

if len(df) > 50000:

    df = df.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

print(f"Training Dataset Shape : {df.shape}")

##########################################################
# Split Features & Target
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
# Models
##########################################################

models = {

    "Decision Tree": DecisionTreeRegressor(

        max_depth=15,

        min_samples_split=10,

        min_samples_leaf=5,

        random_state=42

    ),

    "Random Forest": RandomForestRegressor(

        n_estimators=30,

        max_depth=15,

        min_samples_split=10,

        min_samples_leaf=5,

        random_state=42,

        n_jobs=-1

    ),

    "Gradient Boosting": GradientBoostingRegressor(

        n_estimators=50,

        learning_rate=0.1,

        max_depth=5,

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
    # Model Training
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
    # Evaluation
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
    # Store Results
    ######################################################

    results.append({

        "Model": model_name,

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "R2 Score": round(r2, 4)

    })

    ######################################################
    # Print Metrics
    ######################################################

    print(f"MAE      : {mae:.4f}")

    print(f"RMSE     : {rmse:.4f}")

    print(f"R2 Score : {r2:.4f}")

    ######################################################
    # Best Model
    ######################################################

    if r2 > best_score:

        best_score = r2

        best_model = pipeline

        print("Current Best Tree Model Updated.")

        ##########################################################
# Model Comparison
##########################################################

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 70)
print("TREE MODEL COMPARISON")
print("=" * 70)

print(results_df)

##########################################################
# Save Comparison Report
##########################################################

comparison_path = os.path.join(
    "reports",
    "tree_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

##########################################################
# Save Best Tree Model
##########################################################

best_model_path = os.path.join(
    "models",
    "best_tree_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)

##########################################################
# Best Model Summary
##########################################################

best_model_name = results_df.iloc[0]["Model"]
best_r2 = results_df.iloc[0]["R2 Score"]

print("\n" + "=" * 70)
print("TREE MODEL TRAINING COMPLETED")
print("=" * 70)

print(f"Best Tree Model : {best_model_name}")
print(f"Best R² Score   : {best_r2:.4f}")

print("\nGenerated Files:")

print(f"✓ {comparison_path}")
print(f"✓ {best_model_path}")

print("\nTree Models Ranking:\n")

for i, row in enumerate(results_df.itertuples(index=False), start=1):

    print(
        f"{i}. "
        f"{row.Model} | "
        f"MAE = {row.MAE:.4f} | "
        f"RMSE = {row.RMSE:.4f} | "
        f"R² = {row._3:.4f}"
    )

print("\nTree-based model training completed successfully.")
print("=" * 70)