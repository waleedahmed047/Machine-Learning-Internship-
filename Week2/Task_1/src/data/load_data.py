"""
Week 03 Advanced Regression
Load Dataset Module
"""

import os
import pandas as pd


class DataLoader:
    """
    Loads the dataset from a CSV file.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):

        print("=" * 70)
        print("Loading Dataset...")
        print("=" * 70)

        # Check if file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"\nDataset not found!\nExpected path:\n{self.file_path}"
            )

        # Read dataset
        df = pd.read_csv(self.file_path)

        print(f"Dataset Loaded Successfully")
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        print("\nFirst 5 Rows:")
        print(df.head())

        return df


if __name__ == "__main__":

    loader = DataLoader(
        "data/raw/nyc_taxi_trip_duration.csv"
    )

    dataframe = loader.load()