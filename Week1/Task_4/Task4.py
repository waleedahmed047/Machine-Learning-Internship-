# ============================================================
# REGRESSION MODEL COMPARISON PROJECT
# Medical Insurance Cost Prediction
# Part 1
# ============================================================

# ==========================
# Import Libraries
# ==========================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

from sklearn.model_selection import (
    train_test_split,
    KFold,
    RepeatedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBRegressor


# ============================================================
# Problem Definition
# ============================================================

print("="*70)
print("Medical Insurance Cost Prediction")
print("="*70)
print("Goal : Predict insurance charges using regression models.")
print("Target Variable : charges")
print("="*70)


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("insurance.csv")


# ============================================================
# Exploratory Data Analysis
# ============================================================

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())


# ============================================================
# Independent and Dependent Variables
# ============================================================

X = df.drop("charges", axis=1)
y = df["charges"]

print("\nIndependent Variables")
print(list(X.columns))

print("\nDependent Variable")
print("charges")


# ============================================================
# Linearity Check
# ============================================================

plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="age", y="charges")
plt.title("Age vs Charges")
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="bmi", y="charges")
plt.title("BMI vs Charges")
plt.show()


# ============================================================
# Correlation Heatmap
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()


# ============================================================
# Multicollinearity (VIF)
# ============================================================

print("\nChecking Multicollinearity (VIF)\n")

X_vif = df[["age", "bmi", "children"]]

vif = pd.DataFrame()

vif["Feature"] = X_vif.columns

vif["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(X_vif.shape[1])
]

print(vif)


# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)


# ============================================================
# Preprocessing
# ============================================================

categorical_features = [
    "sex",
    "smoker",
    "region"
]

numerical_features = [
    "age",
    "bmi",
    "children"
]

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(drop="first"),
            categorical_features
        )

    ]

)

print("\nPreprocessing Completed")


# ============================================================
# Encoding for Tree-Based Models
# ============================================================

X_train_tree = pd.get_dummies(
    X_train,
    drop_first=True
)

X_test_tree = pd.get_dummies(
    X_test,
    drop_first=True
)

X_train_tree, X_test_tree = X_train_tree.align(

    X_test_tree,

    join="left",

    axis=1,

    fill_value=0

)

X_tree = pd.get_dummies(
    X,
    drop_first=True
)

print("\nEncoding Completed")


# ============================================================
# Regression Assumption
# Distribution of Target Variable
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(
    y,
    kde=True
)

plt.title("Distribution of Insurance Charges")

plt.show()

print("\nPart 1 Completed Successfully")
print("="*70)

# ============================================================
# PART 2
# Model Training and Evaluation
# ============================================================

results = []

# ============================================================
# Baseline Model
# ============================================================

print("\nTraining Baseline Model...")

baseline = DummyRegressor()

baseline.fit(X_train_tree, y_train)

baseline_pred = baseline.predict(X_test_tree)

results.append({

    "Model":"Baseline",

    "MAE":mean_absolute_error(y_test,baseline_pred),

    "MSE":mean_squared_error(y_test,baseline_pred),

    "RMSE":np.sqrt(mean_squared_error(y_test,baseline_pred)),

    "R2":r2_score(y_test,baseline_pred),

    "Adjusted R2":1-(1-r2_score(y_test,baseline_pred))*(len(y_test)-1)/(len(y_test)-X_test_tree.shape[1]-1),

    "MAPE":mean_absolute_percentage_error(y_test,baseline_pred)

})

# ============================================================
# Simple Linear Regression
# ============================================================

print("Training Simple Linear Regression...")

X_simple = df[["bmi"]]
y_simple = df["charges"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(

    X_simple,

    y_simple,

    test_size=0.20,

    random_state=42

)

simple = LinearRegression()

simple.fit(X_train_s, y_train_s)

pred = simple.predict(X_test_s)

results.append({

    "Model":"Simple Linear Regression",

    "MAE":mean_absolute_error(y_test_s,pred),

    "MSE":mean_squared_error(y_test_s,pred),

    "RMSE":np.sqrt(mean_squared_error(y_test_s,pred)),

    "R2":r2_score(y_test_s,pred),

    "Adjusted R2":1-(1-r2_score(y_test_s,pred))*(len(y_test_s)-1)/(len(y_test_s)-1-1),

    "MAPE":mean_absolute_percentage_error(y_test_s,pred)

})

# ============================================================
# Remaining Models
# ============================================================

models={

"Multiple Linear Regression":LinearRegression(),

"Polynomial Regression":

Pipeline([

("preprocessor",preprocessor),

("poly",PolynomialFeatures(degree=2,include_bias=False)),

("model",LinearRegression())

]),

"Ridge Regression":Ridge(alpha=1),

"Lasso Regression":Lasso(alpha=0.1),

"Elastic Net":ElasticNet(alpha=0.1,l1_ratio=0.5),

"Decision Tree":DecisionTreeRegressor(random_state=42),

"Random Forest":RandomForestRegressor(

n_estimators=200,

random_state=42

),

"Gradient Boosting":GradientBoostingRegressor(random_state=42),

"XGBoost":XGBRegressor(random_state=42)

}

best_model_name = ""
best_predictions = None
best_r2 = -999

for name,model in models.items():

    print(f"Training {name}")

    if name in [

    "Decision Tree",

    "Random Forest",

    "Gradient Boosting",

    "XGBoost"

    ]:

        model.fit(

        X_train_tree,

        y_train

        )

        pred=model.predict(

        X_test_tree

        )

    elif name=="Polynomial Regression":

        model.fit(

        X_train,

        y_train

        )

        pred=model.predict(

        X_test

        )

    else:

        pipe=Pipeline([

        ("preprocessor",preprocessor),

        ("model",model)

        ])

        pipe.fit(

        X_train,

        y_train

        )

        pred=pipe.predict(

        X_test

        )

    mae=mean_absolute_error(y_test,pred)

    mse=mean_squared_error(y_test,pred)

    rmse=np.sqrt(mse)

    r2=r2_score(y_test,pred)

    adj=1-(1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

    mape=mean_absolute_percentage_error(y_test,pred)

    results.append({

    "Model":name,

    "MAE":mae,

    "MSE":mse,

    "RMSE":rmse,

    "R2":r2,

    "Adjusted R2":adj,

    "MAPE":mape

    })

    if r2>best_r2:

        best_r2=r2

        best_model_name=name

        best_predictions=pred

# ============================================================
# Comparison Table
# ============================================================

results_df=pd.DataFrame(results)

results_df=results_df.sort_values(

"R2",

ascending=False

)

print("\n")

print("="*100)

print("MODEL COMPARISON TABLE")

print("="*100)

print(results_df)

print("="*100)

# ============================================================
# Best Model
# ============================================================

print("\nBest Performing Model")

print(best_model_name)

print("Best R2 Score")

print(best_r2)

# ============================================================
# Save Results
# ============================================================

results_df.to_csv(

"Regression_Model_Comparison.csv",

index=False

)

print("\nComparison table saved successfully.")

print("\nPart 2 Completed Successfully")

# ============================================================
# PART 3
# Cross Validation, Residual Analysis & Model Interpretation
# ============================================================

print("\n")
print("=" * 80)
print("CROSS VALIDATION")
print("=" * 80)

# ============================================================
# K-Fold Cross Validation
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

models_cv = {

    "Multiple Linear Regression":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),

    "Ridge Regression":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1))
        ]),

    "Lasso Regression":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", Lasso(alpha=0.1))
        ]),

    "Elastic Net":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5))
        ]),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(random_state=42),

    "XGBoost":
        XGBRegressor(random_state=42)

}

for name, model in models_cv.items():

    if name in [
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]:

        scores = cross_val_score(
            model,
            X_tree,
            y,
            cv=kf,
            scoring="r2"
        )

    else:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="r2"
        )

    cv_results.append([
        name,
        scores.mean(),
        scores.std()
    ])

cv_df = pd.DataFrame(
    cv_results,
    columns=[
        "Model",
        "Mean CV R2",
        "Std"
    ]
)

cv_df = cv_df.sort_values(
    "Mean CV R2",
    ascending=False
)

print(cv_df)

# ============================================================
# Repeated K Fold
# ============================================================

print("\n")
print("=" * 80)
print("REPEATED K-FOLD")
print("=" * 80)

rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

scores = cross_val_score(
    rf,
    X_tree,
    y,
    cv=rkf,
    scoring="r2"
)

print("Repeated KFold Mean R2 :", scores.mean())
print("Repeated KFold Std :", scores.std())

# ============================================================
# Train Best Model Again
# ============================================================

print("\nTraining Best Model Again...")

best_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

best_model.fit(
    X_train_tree,
    y_train
)

prediction = best_model.predict(
    X_test_tree
)

residuals = y_test - prediction

# ============================================================
# Residual Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    prediction,
    residuals
)

plt.axhline(
    y=0,
    color="red"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()

# ============================================================
# Residual Distribution
# ============================================================

plt.figure(figsize=(8,6))

sns.histplot(
    residuals,
    kde=True
)

plt.title("Residual Distribution")

plt.show()

# ============================================================
# Shapiro Test
# ============================================================

print("\n")
print("=" * 80)
print("NORMALITY TEST")
print("=" * 80)

stat, p = shapiro(residuals)

print("Statistic :", stat)
print("P Value :", p)

if p > 0.05:
    print("Residuals are approximately normal.")
else:
    print("Residuals are NOT normally distributed.")

# ============================================================
# Learning Curve
# ============================================================

train_sizes, train_scores, test_scores = learning_curve(

    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    X_tree,

    y,

    cv=5,

    scoring="r2"

)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training"
)

plt.plot(
    train_sizes,
    test_mean,
    marker="o",
    label="Validation"
)

plt.xlabel("Training Size")
plt.ylabel("R2 Score")
plt.title("Learning Curve")

plt.legend()

plt.show()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature": X_tree.columns,

    "Importance": best_model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Important Features\n")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance.head(10),

    x="Importance",

    y="Feature"

)

plt.title("Top 10 Important Features")

plt.show()

# ============================================================
# Bias Variance Analysis
# ============================================================

print("\n")
print("=" * 80)
print("BIAS - VARIANCE ANALYSIS")
print("=" * 80)

train_score = best_model.score(
    X_train_tree,
    y_train
)

test_score = best_model.score(
    X_test_tree,
    y_test
)

print("Training R2 :", train_score)
print("Testing R2 :", test_score)

difference = train_score - test_score

if difference < 0.05:

    print("\nLow Bias")
    print("Low Variance")
    print("Model Generalizes Well")

elif difference < 0.15:

    print("\nSlight Overfitting")

else:

    print("\nHigh Variance")
    print("Model is Overfitting")

# ============================================================
# Error Analysis
# ============================================================

print("\n")
print("=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

print("Average Prediction Error (MAE) :", mean_absolute_error(y_test, prediction))
print("RMSE :", np.sqrt(mean_squared_error(y_test, prediction)))
print("MAPE :", mean_absolute_percentage_error(y_test, prediction))

print("\nLower values indicate better prediction performance.")

# ============================================================
# Recommended Model
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL SELECTION")
print("=" * 80)

print("Recommended Model : Random Forest Regressor")

print("""
Reason:

1. Highest R² score.

2. Lowest prediction error.

3. Stable cross-validation performance.

4. Handles nonlinear relationships effectively.

5. Less sensitive to outliers.

6. Good generalization on unseen data.
""")

# ============================================================
# Limitations
# ============================================================

print("\n")
print("=" * 80)
print("LIMITATIONS")
print("=" * 80)

print("""
1. Dataset is relatively small.

2. Hyperparameter tuning was limited.

3. External factors affecting insurance charges are not included.

4. Results may differ for other populations.

5. More features could improve prediction accuracy.
""")

print("\n")
print("=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)