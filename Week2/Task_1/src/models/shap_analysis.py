"""
Week 02 Internship Project
SHAP Explainability Analysis
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.features.build_features import FeatureBuilder

print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

##########################################################
# Create Folder
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
# Fast Sample
##########################################################

if len(df) > 50000:

    df = df.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

##########################################################
# Target
##########################################################

TARGET = "trip_duration"

X = df.drop(columns=[TARGET])

##########################################################
# Feature Engineering
##########################################################

builder = FeatureBuilder()

X = builder.fit_transform(X)

##########################################################
# Load Model
##########################################################

print("\nLoading Tuned Model...")

pipeline = joblib.load(
    "models/tuned_random_forest.joblib"
)

##########################################################
# Separate Preprocessor & Model
##########################################################

preprocessor = pipeline.named_steps["preprocessor"]

model = pipeline.named_steps["model"]

##########################################################
# Transform Features
##########################################################

X_processed = preprocessor.transform(X)

try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"Feature_{i}" for i in range(X_processed.shape[1])]

##########################################################
# SHAP Sample
##########################################################

sample_size = min(
    300,
    X_processed.shape[0]
)

X_sample = X_processed[:sample_size]

print(f"\nSHAP Sample Size : {sample_size}")

##########################################################
# Create Explainer
##########################################################

explainer = shap.TreeExplainer(model)

print("Calculating SHAP Values...")

shap_values = explainer.shap_values(X_sample)

print("SHAP Values Calculated Successfully.")

##########################################################
# SHAP Summary Plot
##########################################################

print("\nGenerating SHAP Summary Plot...")


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    shap_values,

    X_sample,

    feature_names=feature_names,

    show=False

)


plt.title(
    "SHAP Feature Impact Summary"
)


plt.tight_layout()


plt.savefig(

    "reports/figures/shap_summary_plot.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


##########################################################
# SHAP Feature Importance Bar Plot
##########################################################

print("\nGenerating SHAP Importance Plot...")


plt.figure(

    figsize=(10, 8)

)


shap.summary_plot(

    shap_values,

    X_sample,

    feature_names=feature_names,

    plot_type="bar",

    show=False

)


plt.title(

    "SHAP Feature Importance"

)


plt.tight_layout()


plt.savefig(

    "reports/figures/shap_feature_importance.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("\nSHAP Summary Graphs Generated Successfully.")

##########################################################
# SHAP Waterfall Plot
##########################################################

print("\nGenerating SHAP Waterfall Plot...")


try:

    explanation = shap.Explanation(

        values=shap_values[0],

        base_values=explainer.expected_value,

        data=X_sample[0],

        feature_names=feature_names

    )


    plt.figure(

        figsize=(12, 8)

    )


    shap.plots.waterfall(

        explanation,

        show=False

    )


    plt.tight_layout()


    plt.savefig(

        "reports/figures/shap_waterfall_plot.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print("Waterfall Plot Generated Successfully.")


except Exception as e:

    print(

        "Waterfall plot skipped:",

        e

    )


##########################################################
# Save SHAP Values
##########################################################

print("\nSaving SHAP Values...")


shap_df = pd.DataFrame(

    shap_values,

    columns=feature_names

)


shap_df.to_csv(

    "reports/shap_values.csv",

    index=False

)


##########################################################
# Feature Importance Table
##########################################################

print("\nCreating Feature Importance Report...")


importance = pd.DataFrame({

    "Feature":

        feature_names,


    "Mean_SHAP_Value":

        abs(shap_values).mean(axis=0)

})


importance = importance.sort_values(

    by="Mean_SHAP_Value",

    ascending=False

)


importance.to_csv(

    "reports/shap_feature_importance.csv",

    index=False

)


##########################################################
# Final Summary
##########################################################

print("\n" + "=" * 70)

print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")

print("=" * 70)


print("\nGenerated Files:")


files = [

    "reports/figures/shap_summary_plot.png",

    "reports/figures/shap_feature_importance.png",

    "reports/figures/shap_waterfall_plot.png",

    "reports/shap_values.csv",

    "reports/shap_feature_importance.csv"

]


for file in files:

    print(f"✓ {file}")


print("\nTop Important Features:")


print(

    importance.head(10)

)


print("\n" + "=" * 70)

print("MODEL EXPLAINABILITY COMPLETED")

print("=" * 70)
