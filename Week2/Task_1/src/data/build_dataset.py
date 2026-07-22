"""
Week 03 Advanced Regression
Build Clean Dataset
"""

import os
import pandas as pd

from src.data.load_data import DataLoader
from src.data.validate_data import DataValidator


def clean_dataset(df):
    """
    Clean dataset by removing duplicates
    and handling missing values.
    """

    print("\n" + "=" * 70)
    print("CLEANING DATASET")
    print("=" * 70)

    # Remove duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows...")
        df = df.drop_duplicates()

    # Missing Values
    missing = df.isnull().sum().sum()

    if missing > 0:
        print(f"Handling {missing} missing values...")

        # Numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns

        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())

        # Object columns
        object_columns = df.select_dtypes(include=["object"]).columns

        for col in object_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nCleaning Completed Successfully.")

    print(f"Final Shape : {df.shape}")

    return df


def save_dataset(df):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = "data/processed/clean_data.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nDataset saved at:\n{output_path}")


def main():

    loader = DataLoader(
        "data/raw/nyc_taxi_trip_duration.csv"
    )

    df = loader.load()

    validator = DataValidator(df)

    validator.run()

    clean_df = clean_dataset(df)

    save_dataset(clean_df)


if __name__ == "__main__":
    main()