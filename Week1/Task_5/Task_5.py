# ==========================================================
# Machine Learning Internship - Week 2
# Part 1: Data Quality & Data Cleaning
# Dataset: Titanic (train.csv)
# ==========================================================

import pandas as pd
import numpy as np

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("train.csv")

# ----------------------------------------------------------
# 2. Basic Information
# ----------------------------------------------------------

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe(include="all"))

# ----------------------------------------------------------
# 3. Missing Values
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

missing = df.isnull().sum()

missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percent
})

print(missing_table)

# ----------------------------------------------------------
# 4. Duplicate Records
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Duplicate Records")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate Rows :", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates Removed.")
else:
    print("No Duplicate Records Found.")

# ----------------------------------------------------------
# 5. Incorrect Data Types
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Incorrect Data Types")
print("=" * 60)

print(df.dtypes)

# Example Conversion
# PassengerId should be integer
df["PassengerId"] = df["PassengerId"].astype(int)

# Age should be float
df["Age"] = df["Age"].astype(float)

print("\nUpdated Data Types")
print(df.dtypes)

# ----------------------------------------------------------
# 6. Invalid Values
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Checking Invalid Values")
print("=" * 60)

# Age cannot be negative

invalid_age = df[df["Age"] < 0]

print("Negative Ages Found :", len(invalid_age))

# Fare cannot be negative

invalid_fare = df[df["Fare"] < 0]

print("Negative Fare Found :", len(invalid_fare))

# ----------------------------------------------------------
# 7. Inconsistent Categories
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Categorical Values")
print("=" * 60)

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:

    print(f"\nColumn : {col}")

    print(df[col].value_counts(dropna=False))

# Example Cleaning

df["Sex"] = df["Sex"].str.lower()

df["Embarked"] = df["Embarked"].str.upper()

print("\nAfter Cleaning")

print(df["Sex"].unique())

print(df["Embarked"].unique())

# ----------------------------------------------------------
# 8. Noisy Data
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Noise Handling")
print("=" * 60)

# Round Age values

df["Age"] = df["Age"].round()

print(df["Age"].head())

# ----------------------------------------------------------
# 9. Outlier Detection (IQR Method)
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Outlier Detection")
print("=" * 60)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:

    if df[col].isnull().sum() > 0:
        continue

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col:15} -> {len(outliers)} Outliers")

# ----------------------------------------------------------
# 10. Remove Extreme Outliers (Fare Example)
# ----------------------------------------------------------

print("\nRemoving Extreme Fare Outliers")

Q1 = df["Fare"].quantile(0.25)

Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

before = len(df)

df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

after = len(df)

print("Rows Before :", before)

print("Rows After  :", after)

print("Removed     :", before - after)

# ----------------------------------------------------------
# 11. Final Dataset Information
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Final Dataset")
print("=" * 60)

print(df.head())

print("\nShape :", df.shape)

print("\nMissing Values")

print(df.isnull().sum())

print("\nData Types")

print(df.dtypes)

# ----------------------------------------------------------
# 12. Save Clean Dataset
# ----------------------------------------------------------

df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

print("\nOutput File : Titanic_Cleaned.csv")

# ==========================================================
# PART 2 : Missing Value Treatment
# ==========================================================

print("\n" + "="*60)
print("PART 2 : Missing Value Treatment")
print("="*60)

# Reload original dataset
df = pd.read_csv("train.csv")

# -------------------------------
# Missing Values Before Treatment
# -------------------------------

print("\nMissing Values Before Treatment")

print(df.isnull().sum())

# ==========================================================
# Method 1 : Row Deletion
# ==========================================================

print("\n" + "="*60)
print("1. Row Deletion")
print("="*60)

row_deleted = df.dropna()

print("Original Shape :", df.shape)
print("After Row Deletion :", row_deleted.shape)

# ==========================================================
# Method 2 : Column Deletion
# ==========================================================

print("\n" + "="*60)
print("2. Column Deletion")
print("="*60)

column_deleted = df.copy()

threshold = len(column_deleted) * 0.50

column_deleted = column_deleted.dropna(
    axis=1,
    thresh=threshold
)

print("Remaining Columns")

print(column_deleted.columns.tolist())

# ==========================================================
# Method 3 : Mean Imputation
# ==========================================================

print("\n" + "="*60)
print("3. Mean Imputation")
print("="*60)

mean_df = df.copy()

numeric_columns = mean_df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    if mean_df[col].isnull().sum() > 0:

        mean_df[col].fillna(
            mean_df[col].mean(),
            inplace=True
        )

print(mean_df.isnull().sum())

# ==========================================================
# Method 4 : Median Imputation
# ==========================================================

print("\n" + "="*60)
print("4. Median Imputation")
print("="*60)

median_df = df.copy()

numeric_columns = median_df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    if median_df[col].isnull().sum() > 0:

        median_df[col].fillna(
            median_df[col].median(),
            inplace=True
        )

print(median_df.isnull().sum())

# ==========================================================
# Method 5 : Mode Imputation
# ==========================================================

print("\n" + "="*60)
print("5. Mode Imputation")
print("="*60)

mode_df = df.copy()

for col in mode_df.columns:

    if mode_df[col].isnull().sum() > 0:

        mode_df[col].fillna(
            mode_df[col].mode()[0],
            inplace=True
        )

print(mode_df.isnull().sum())

# ==========================================================
# Method 6 : SimpleImputer
# ==========================================================

print("\n" + "="*60)
print("6. SimpleImputer")
print("="*60)

from sklearn.impute import SimpleImputer

simple_df = df.copy()

num_cols = simple_df.select_dtypes(include=np.number).columns

cat_cols = simple_df.select_dtypes(include="object").columns

num_imputer = SimpleImputer(strategy="median")

cat_imputer = SimpleImputer(strategy="most_frequent")

simple_df[num_cols] = num_imputer.fit_transform(simple_df[num_cols])

simple_df[cat_cols] = cat_imputer.fit_transform(simple_df[cat_cols])

print(simple_df.isnull().sum())

# ==========================================================
# Method 7 : KNN Imputer (Advanced)
# ==========================================================

print("\n" + "="*60)
print("7. KNN Imputation")
print("="*60)

from sklearn.impute import KNNImputer

knn_df = df.copy()

knn_numeric = knn_df.select_dtypes(include=np.number).columns

knn = KNNImputer(n_neighbors=5)

knn_df[knn_numeric] = knn.fit_transform(knn_df[knn_numeric])

print(knn_df[knn_numeric].isnull().sum())

# ==========================================================
# Compare All Imputation Methods
# ==========================================================

print("\n" + "="*60)
print("Comparison of Imputation Methods")
print("="*60)

comparison = pd.DataFrame({

    "Original Missing": df.isnull().sum(),

    "Mean": mean_df.isnull().sum(),

    "Median": median_df.isnull().sum(),

    "Mode": mode_df.isnull().sum(),

    "SimpleImputer": simple_df.isnull().sum()

})

print(comparison)

# ==========================================================
# Save Best Dataset
# ==========================================================

simple_df.to_csv(
    "Titanic_After_Imputation.csv",
    index=False
)

print("\nDataset Saved Successfully")

print("File Name : Titanic_After_Imputation.csv")

# ==========================================================
# PART 3 : Categorical Data Encoding
# ==========================================================

print("\n" + "="*60)
print("PART 3 : Categorical Data Encoding")
print("="*60)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Load dataset after missing value treatment
df = pd.read_csv("Titanic_After_Imputation.csv")

print("\nCategorical Columns")

categorical_columns = df.select_dtypes(include="object").columns

print(categorical_columns.tolist())

# ==========================================================
# Label Encoding
# ==========================================================

print("\n" + "="*60)
print("1. Label Encoding")
print("="*60)

label_df = df.copy()

le = LabelEncoder()

label_columns = ["Sex"]

for col in label_columns:

    label_df[col] = le.fit_transform(label_df[col])

print(label_df[label_columns].head())

# ==========================================================
# Ordinal Encoding
# ==========================================================

print("\n" + "="*60)
print("2. Ordinal Encoding")
print("="*60)

ordinal_df = label_df.copy()

# Create an ordinal feature from Age

ordinal_df["Age_Group"] = pd.cut(
    ordinal_df["Age"],
    bins=[0,12,19,35,60,100],
    labels=["Child","Teen","Young","Adult","Senior"]
)

encoder = OrdinalEncoder(
    categories=[["Child","Teen","Young","Adult","Senior"]]
)

ordinal_df["Age_Group"] = encoder.fit_transform(
    ordinal_df[["Age_Group"]]
)

print(ordinal_df[["Age","Age_Group"]].head())

# ==========================================================
# One Hot Encoding
# ==========================================================

print("\n" + "="*60)
print("3. One Hot Encoding")
print("="*60)

onehot_df = ordinal_df.copy()

onehot_columns = ["Embarked"]

onehot = pd.get_dummies(
    onehot_df,
    columns=onehot_columns,
    drop_first=True,
    dtype=int
)

print(onehot.head())

# ==========================================================
# High Cardinality Handling
# ==========================================================

print("\n" + "="*60)
print("4. High Cardinality")
print("="*60)

high_df = onehot.copy()

# Cabin has many unique values

if "Cabin" in high_df.columns:

    print("Unique Cabins :", high_df["Cabin"].nunique())

    top10 = high_df["Cabin"].value_counts().nlargest(10).index

    high_df["Cabin"] = np.where(
        high_df["Cabin"].isin(top10),
        high_df["Cabin"],
        "Other"
    )

    print(high_df["Cabin"].value_counts())

# ==========================================================
# Rare Category Handling
# ==========================================================

print("\n" + "="*60)
print("5. Rare Category Handling")
print("="*60)

rare_df = high_df.copy()

if "Cabin" in rare_df.columns:

    counts = rare_df["Cabin"].value_counts()

    rare_categories = counts[counts < 5].index

    rare_df["Cabin"] = rare_df["Cabin"].replace(
        rare_categories,
        "Rare"
    )

    print(rare_df["Cabin"].value_counts())

# ==========================================================
# Frequency Encoding (Optional)
# ==========================================================

print("\n" + "="*60)
print("6. Frequency Encoding")
print("="*60)

freq_df = rare_df.copy()

if "Ticket" in freq_df.columns:

    frequency = freq_df["Ticket"].value_counts()

    freq_df["Ticket_Frequency"] = freq_df["Ticket"].map(frequency)

    print(freq_df[["Ticket","Ticket_Frequency"]].head())

# ==========================================================
# Target Encoding Demonstration (Educational)
# ==========================================================

print("\n" + "="*60)
print("7. Target Mean Encoding (Example)")
print("="*60)

target_df = freq_df.copy()

if "Embarked_S" in target_df.columns:

    print("Target Encoding skipped because One-Hot Encoding is already applied.")
    print("Normally this technique is used for high-cardinality variables.")

# ==========================================================
# Final Encoded Dataset
# ==========================================================

print("\nEncoded Dataset Shape")

print(freq_df.shape)

print("\nEncoded Dataset Columns")

print(freq_df.columns.tolist())

# ==========================================================
# Save Dataset
# ==========================================================

freq_df.to_csv(
    "Titanic_Encoded.csv",
    index=False
)

print("\nEncoded Dataset Saved Successfully!")

print("Output File : Titanic_Encoded.csv")

# ==========================================================
# PART 4 : Numerical Data & Feature Engineering
# ==========================================================

print("\n" + "="*60)
print("PART 4 : Numerical Data & Feature Engineering")
print("="*60)

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------
# Load Encoded Dataset
# ----------------------------------------------------------

df = pd.read_csv("Titanic_Encoded.csv")

print("\nDataset Shape :", df.shape)

# ==========================================================
# Numerical Columns
# ==========================================================

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumerical Columns")

print(numerical_columns)

# ==========================================================
# Standardization
# ==========================================================

print("\n" + "="*60)
print("1. StandardScaler")
print("="*60)

standard_df = df.copy()

standard_scaler = StandardScaler()

scale_columns = ["Age","Fare"]

standard_df[scale_columns] = standard_scaler.fit_transform(
    standard_df[scale_columns]
)

print(standard_df[scale_columns].head())

# ==========================================================
# Min-Max Scaling
# ==========================================================

print("\n" + "="*60)
print("2. MinMaxScaler")
print("="*60)

minmax_df = df.copy()

minmax_scaler = MinMaxScaler()

minmax_df[scale_columns] = minmax_scaler.fit_transform(
    minmax_df[scale_columns]
)

print(minmax_df[scale_columns].head())

# ==========================================================
# Robust Scaling
# ==========================================================

print("\n" + "="*60)
print("3. RobustScaler")
print("="*60)

robust_df = df.copy()

robust_scaler = RobustScaler()

robust_df[scale_columns] = robust_scaler.fit_transform(
    robust_df[scale_columns]
)

print(robust_df[scale_columns].head())

# ==========================================================
# Log Transformation
# ==========================================================

print("\n" + "="*60)
print("4. Log Transformation")
print("="*60)

log_df = df.copy()

log_df["Fare_Log"] = np.log1p(log_df["Fare"])

print(log_df[["Fare","Fare_Log"]].head())

# ==========================================================
# Outlier Treatment
# ==========================================================

print("\n" + "="*60)
print("5. Outlier Treatment")
print("="*60)

outlier_df = df.copy()

Q1 = outlier_df["Fare"].quantile(0.25)

Q3 = outlier_df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

before = len(outlier_df)

outlier_df["Fare"] = outlier_df["Fare"].clip(lower, upper)

after = len(outlier_df)

print("Rows Before :", before)
print("Rows After  :", after)

# ==========================================================
# Feature Engineering
# ==========================================================

print("\n" + "="*60)
print("Feature Engineering")
print("="*60)

feature_df = outlier_df.copy()

# ----------------------------------------------------------
# Derived Variable
# ----------------------------------------------------------

feature_df["FamilySize"] = (
    feature_df["SibSp"] +
    feature_df["Parch"] +
    1
)

print("\nFamilySize Created")

# ----------------------------------------------------------
# Domain Feature
# ----------------------------------------------------------

feature_df["IsAlone"] = np.where(
    feature_df["FamilySize"] == 1,
    1,
    0
)

print("IsAlone Feature Created")

# ----------------------------------------------------------
# Text Length Feature
# ----------------------------------------------------------

feature_df["Name_Length"] = feature_df["Name"].str.len()

print("Name_Length Feature Created")

# ----------------------------------------------------------
# Ticket Length
# ----------------------------------------------------------

feature_df["Ticket_Length"] = feature_df["Ticket"].astype(str).str.len()

print("Ticket_Length Feature Created")

# ----------------------------------------------------------
# Cabin Availability
# ----------------------------------------------------------

if "Cabin" in feature_df.columns:

    feature_df["HasCabin"] = np.where(
        feature_df["Cabin"].isnull(),
        0,
        1
    )

print("HasCabin Feature Created")

# ----------------------------------------------------------
# Ratio Feature
# ----------------------------------------------------------

feature_df["Fare_Per_Person"] = (
    feature_df["Fare"] /
    feature_df["FamilySize"]
)

print("Fare_Per_Person Created")

# ----------------------------------------------------------
# Age Binning
# ----------------------------------------------------------

feature_df["AgeGroup"] = pd.cut(

    feature_df["Age"],

    bins=[0,12,19,35,60,100],

    labels=[
        "Child",
        "Teen",
        "Young",
        "Adult",
        "Senior"
    ]

)

print("AgeGroup Created")

# ----------------------------------------------------------
# Interaction Feature
# ----------------------------------------------------------

feature_df["Age_Fare"] = (

    feature_df["Age"] *

    feature_df["Fare"]

)

print("Age_Fare Created")

# ----------------------------------------------------------
# Passenger Title
# ----------------------------------------------------------

feature_df["Title"] = (

    feature_df["Name"]

    .str.extract(" ([A-Za-z]+)\.", expand=False)

)

print("Title Feature Created")

# ----------------------------------------------------------
# Rare Titles
# ----------------------------------------------------------

rare_titles = [

    "Lady","Countess","Capt","Col","Don",

    "Dr","Major","Rev","Sir",

    "Jonkheer","Dona"

]

feature_df["Title"] = feature_df["Title"].replace(

    rare_titles,

    "Rare"

)

feature_df["Title"] = feature_df["Title"].replace({

    "Mlle":"Miss",

    "Ms":"Miss",

    "Mme":"Mrs"

})

print("Title Cleaned")

# ----------------------------------------------------------
# Age Category
# ----------------------------------------------------------

feature_df["Age_Category"] = np.where(

    feature_df["Age"] < 18,

    "Child",

    "Adult"

)

# ----------------------------------------------------------
# Display New Features
# ----------------------------------------------------------

print("\nNew Features")

new_columns = [

    "FamilySize",

    "IsAlone",

    "Name_Length",

    "Ticket_Length",

    "Fare_Per_Person",

    "Age_Fare",

    "Title",

    "Age_Category"

]

print(feature_df[new_columns].head())

# ----------------------------------------------------------
# Save Dataset
# ----------------------------------------------------------

feature_df.to_csv(

    "Titanic_Feature_Engineered.csv",

    index=False

)

print("\nFeature Engineered Dataset Saved")

print("Output : Titanic_Feature_Engineered.csv")

# ==========================================================
# PART 5 : Data Splitting, Data Leakage & Pipeline
# ==========================================================

print("\n" + "="*60)
print("PART 5 : Data Splitting, Data Leakage & Pipeline")
print("="*60)

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    TimeSeriesSplit
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

# ----------------------------------------------------------
# Load Feature Engineered Dataset
# ----------------------------------------------------------

df = pd.read_csv("Titanic_Feature_Engineered.csv")

print("\nDataset Shape")

print(df.shape)

# ----------------------------------------------------------
# Define Features and Target
# ----------------------------------------------------------

X = df.drop(columns=["Survived"])

y = df["Survived"]

# ==========================================================
# Random Split
# ==========================================================

print("\n" + "="*60)
print("1. Random Train/Test Split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("Training Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

# ==========================================================
# Validation Split
# ==========================================================

print("\n" + "="*60)
print("2. Validation Split")
print("="*60)

from sklearn.model_selection import train_test_split

# First split: Train + Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Second split: Train + Validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_valid.shape)
print("Test:", X_test.shape)

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_valid.shape)
print("Testing Shape :", X_test.shape)

# ==========================================================
# Stratified Split
# ==========================================================

print("\n" + "="*60)
print("3. Stratified Split")
print("="*60)

sss = StratifiedShuffleSplit(

    n_splits=1,

    test_size=0.20,

    random_state=42

)

for train_index, test_index in sss.split(X, y):

    X_train_strat = X.iloc[train_index]

    X_test_strat = X.iloc[test_index]

    y_train_strat = y.iloc[train_index]

    y_test_strat = y.iloc[test_index]

print("Training :", X_train_strat.shape)

print("Testing :", X_test_strat.shape)

# ==========================================================
# Time Based Split Example
# ==========================================================

print("\n" + "="*60)
print("4. Time Based Split Example")
print("="*60)

time_df = df.copy()

time_df["OrderDate"] = pd.date_range(

    start="2024-01-01",

    periods=len(time_df),

    freq="D"

)

time_df = time_df.sort_values("OrderDate")

tscv = TimeSeriesSplit(n_splits=5)

for i, (train, test) in enumerate(tscv.split(time_df)):

    print(f"Split {i+1}")

    print("Train:", len(train))

    print("Test :", len(test))

# ==========================================================
# Data Leakage
# ==========================================================

print("\n" + "="*60)
print("5. Data Leakage")
print("="*60)

print("""
Target Leakage:
Using information that would not be available
at prediction time.

Example:
Using 'Survived' while creating features.

--------------------------------------------

Training/Test Contamination:
Applying preprocessing on the entire dataset
before splitting.

Wrong:

Scaler.fit(all_data)

Correct:

Scaler.fit(training_data)

--------------------------------------------

Future Information Leakage:

Using future dates or future values while
training the model.
""")

# ==========================================================
# Pipeline
# ==========================================================

print("\n" + "="*60)
print("6. Scikit-Learn Pipeline")
print("="*60)

numeric_features = X_train.select_dtypes(

    include=np.number

).columns.tolist()

categorical_features = X_train.select_dtypes(

    exclude=np.number

).columns.tolist()

print("\nNumeric Columns")

print(numeric_features)

print("\nCategorical Columns")

print(categorical_features)

# Numeric Pipeline

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

# Categorical Pipeline

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

# Column Transformer

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_pipeline,

            numeric_features

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_features

        )

    ]

)

print("\nPipeline Created Successfully")

# ==========================================================
# Fit Pipeline
# ==========================================================

X_train_processed = preprocessor.fit_transform(

    X_train

)

X_test_processed = preprocessor.transform(

    X_test

)

print("\nProcessed Training Shape")

print(X_train_processed.shape)

print("\nProcessed Testing Shape")

print(X_test_processed.shape)

# ==========================================================
# Reproducibility
# ==========================================================

print("\n" + "="*60)
print("7. Reproducibility")
print("="*60)

RANDOM_STATE = 42

print("Random State =", RANDOM_STATE)

print("Using fixed random_state ensures")

print("same results every time.")

# ==========================================================
# Save Processed Data
# ==========================================================

processed_train = pd.DataFrame(

    X_train_processed.toarray()

    if hasattr(X_train_processed, "toarray")

    else X_train_processed

)

processed_train.to_csv(

    "Titanic_Final_Preprocessed.csv",

    index=False

)

print("\nFinal Preprocessed Dataset Saved")

print("Output File : Titanic_Final_Preprocessed.csv")

print("\nAssignment Completed Successfully!")

# ==========================================================
# PART 6 : Boston Housing Dataset
# ==========================================================

print("\n" + "="*70)
print("PART 6 : Boston Housing Dataset")
print("="*70)

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

boston = pd.read_csv("boston.csv")

print("\nFirst Five Rows")
print(boston.head())

print("\nDataset Shape")
print(boston.shape)

print("\nMissing Values")
print(boston.isnull().sum())

# ----------------------------------------------------------
# Remove Duplicates
# ----------------------------------------------------------

duplicates = boston.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

boston.drop_duplicates(inplace=True)

# ----------------------------------------------------------
# Data Types
# ----------------------------------------------------------

print("\nData Types")

print(boston.dtypes)

# ----------------------------------------------------------
# Missing Value Treatment
# ----------------------------------------------------------

num_cols = boston.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy="median")

boston[num_cols] = imputer.fit_transform(boston[num_cols])

# ----------------------------------------------------------
# Outlier Detection
# ----------------------------------------------------------

print("\nOutlier Detection")

for col in num_cols:

    Q1 = boston[col].quantile(0.25)

    Q3 = boston[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR

    outliers = ((boston[col] < lower) |
                (boston[col] > upper)).sum()

    print(col, ":", outliers)

# ----------------------------------------------------------
# Outlier Treatment
# ----------------------------------------------------------

for col in num_cols:

    Q1 = boston[col].quantile(0.25)

    Q3 = boston[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR

    boston[col] = boston[col].clip(lower, upper)

# ----------------------------------------------------------
# Scaling
# ----------------------------------------------------------

standard = StandardScaler()

minmax = MinMaxScaler()

robust = RobustScaler()

standard_data = standard.fit_transform(boston[num_cols])

minmax_data = minmax.fit_transform(boston[num_cols])

robust_data = robust.fit_transform(boston[num_cols])

print("\nScaling Completed")

# ----------------------------------------------------------
# Log Transformation
# ----------------------------------------------------------

if "CRIM" in boston.columns:

    boston["CRIM_LOG"] = np.log1p(boston["CRIM"])

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------

if "RM" in boston.columns:

    boston["Rooms_Per_Age"] = boston["RM"] / (boston["AGE"] + 1)

if "DIS" in boston.columns:

    boston["Accessibility"] = boston["DIS"] * boston["RAD"]

if "LSTAT" in boston.columns:

    boston["Income_Category"] = pd.cut(

        boston["LSTAT"],

        bins=[0,10,20,40],

        labels=["High","Medium","Low"]

    )

# ----------------------------------------------------------
# Data Splitting
# ----------------------------------------------------------

target = "MEDV"

X = boston.drop(columns=[target])

y = boston[target]

X_train,X_test,y_train,y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

print("\nTraining Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)

# ----------------------------------------------------------
# Pipeline
# ----------------------------------------------------------

numeric_features = X_train.select_dtypes(include=np.number).columns

categorical_features = X_train.select_dtypes(exclude=np.number).columns

numeric_pipeline = Pipeline([

    ("imputer",SimpleImputer(strategy="median")),

    ("scaler",StandardScaler())

])

categorical_pipeline = Pipeline([

    ("imputer",SimpleImputer(strategy="most_frequent"))

])

preprocessor = ColumnTransformer([

    ("num",numeric_pipeline,numeric_features),

    ("cat",categorical_pipeline,categorical_features)

])

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nPipeline Completed")

print("Processed Training Shape :", X_train_processed.shape)

print("Processed Testing Shape :", X_test_processed.shape)

# ----------------------------------------------------------
# Save Dataset
# ----------------------------------------------------------

boston.to_csv(

    "Boston_Preprocessed.csv",

    index=False

)

print("\nBoston Dataset Saved Successfully!")

print("Output : Boston_Preprocessed.csv")