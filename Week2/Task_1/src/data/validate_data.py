"""
Week 03 Advanced Regression
Data Validation Module
"""

import pandas as pd


class DataValidator:

    def __init__(self, dataframe):
        self.df = dataframe

    ########################################################

    def dataset_shape(self):

        print("\n" + "=" * 70)
        print("DATASET SHAPE")
        print("=" * 70)

        print(self.df.shape)

    ########################################################

    def column_information(self):

        print("\n" + "=" * 70)
        print("COLUMNS")
        print("=" * 70)

        for column in self.df.columns:
            print(column)

    ########################################################

    def data_types(self):

        print("\n" + "=" * 70)
        print("DATA TYPES")
        print("=" * 70)

        print(self.df.dtypes)

    ########################################################

    def missing_values(self):

        print("\n" + "=" * 70)
        print("MISSING VALUES")
        print("=" * 70)

        print(self.df.isnull().sum())

    ########################################################

    def duplicate_records(self):

        print("\n" + "=" * 70)
        print("DUPLICATE RECORDS")
        print("=" * 70)

        print(self.df.duplicated().sum())

    ########################################################

    def numerical_summary(self):

        print("\n" + "=" * 70)
        print("NUMERICAL SUMMARY")
        print("=" * 70)

        print(self.df.describe())

    ########################################################

    def categorical_summary(self):

        print("\n" + "=" * 70)
        print("CATEGORICAL SUMMARY")
        print("=" * 70)

        try:
            print(self.df.describe(include=["object"]))
        except Exception:
            print("No categorical columns found.")

    ########################################################

    def run(self):

        self.dataset_shape()

        self.column_information()

        self.data_types()

        self.missing_values()

        self.duplicate_records()

        self.numerical_summary()

        self.categorical_summary()


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/nyc_taxi_trip_duration.csv"
    )

    validator = DataValidator(df)

    validator.run()