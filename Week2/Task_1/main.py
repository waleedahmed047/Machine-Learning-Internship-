"""
Week 03 Advanced Regression
Main File
"""

from src.data.load_data import DataLoader
from src.data.validate_data import DataValidator
from src.data.build_dataset import clean_dataset


def main():

    print("=" * 70)
    print("NYC Taxi Trip Duration Prediction Project")
    print("=" * 70)

    loader = DataLoader(
        "data/raw/nyc_taxi_trip_duration.csv"
    )

    df = loader.load()

    validator = DataValidator(df)

    validator.run()

    clean_df = clean_dataset(df)

    clean_df.to_csv(
        "data/processed/clean_data.csv",
        index=False
    )

    print("\nDataset cleaned successfully.")

    print("\nNext Step:")
    print("Run:")
    print("python src/models/train.py")


if __name__ == "__main__":
    main()