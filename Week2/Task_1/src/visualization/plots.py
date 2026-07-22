"""
Week 03 Advanced Regression
Exploratory Data Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

FIGURE_PATH = "reports/figures"

os.makedirs(FIGURE_PATH, exist_ok=True)


class EDA:

    def __init__(self, df):

        self.df = df.copy()

    ########################################################

    def create_target(self):

        if (
            "pickup_datetime" in self.df.columns
            and
            "dropoff_datetime" in self.df.columns
        ):

            self.df["pickup_datetime"] = pd.to_datetime(
                self.df["pickup_datetime"]
            )

            self.df["dropoff_datetime"] = pd.to_datetime(
                self.df["dropoff_datetime"]
            )

            self.df["trip_duration"] = (
                self.df["dropoff_datetime"]
                -
                self.df["pickup_datetime"]
            ).dt.total_seconds() / 60

    ########################################################

    def duration_distribution(self):

        if "trip_duration" not in self.df.columns:
            return

        plt.figure(figsize=(10,5))

        sns.histplot(
            self.df["trip_duration"],
            bins=50,
            kde=True
        )

        plt.title("Trip Duration Distribution")

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/duration_distribution.png"
        )

        plt.close()

    ########################################################

    def log_distribution(self):

        if "trip_duration" not in self.df.columns:
            return

        plt.figure(figsize=(10,5))

        sns.histplot(
            np.log1p(self.df["trip_duration"]),
            bins=50,
            kde=True
        )

        plt.title("Log Trip Duration")

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/log_duration.png"
        )

        plt.close()

    ########################################################

    def duration_by_hour(self):

        if "pickup_datetime" not in self.df.columns:
            return

        self.df["hour"] = (
            self.df["pickup_datetime"].dt.hour
        )

        plt.figure(figsize=(12,6))

        sns.boxplot(
            x="hour",
            y="trip_duration",
            data=self.df
        )

        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/duration_by_hour.png"
        )

        plt.close()

    ########################################################

    def duration_by_day(self):

        if "pickup_datetime" not in self.df.columns:
            return

        self.df["weekday"] = (
            self.df["pickup_datetime"]
            .dt.day_name()
        )

        plt.figure(figsize=(12,6))

        sns.boxplot(
            x="weekday",
            y="trip_duration",
            data=self.df
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/duration_by_day.png"
        )

        plt.close()

    ########################################################

    def passenger_count(self):

        if (
            "passenger_count" not in self.df.columns
            or
            "trip_duration" not in self.df.columns
        ):
            return

        plt.figure(figsize=(10,5))

        sns.boxplot(
            x="passenger_count",
            y="trip_duration",
            data=self.df
        )

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/passenger_count.png"
        )

        plt.close()

    ########################################################

    def correlation_matrix(self):

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if numeric.empty:
            return

        plt.figure(figsize=(12,10))

        sns.heatmap(
            numeric.corr(),
            cmap="coolwarm",
            center=0
        )

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/correlation_matrix.png"
        )

        plt.close()

    ########################################################

    def missing_values(self):

        missing = (
            self.df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10,6))

        missing.plot(kind="bar")

        plt.title("Missing Values")

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/missing_values.png"
        )

        plt.close()

    ########################################################

    def outliers(self):

        if "trip_duration" not in self.df.columns:
            return

        plt.figure(figsize=(8,6))

        sns.boxplot(
            y=self.df["trip_duration"]
        )

        plt.tight_layout()

        plt.savefig(
            f"{FIGURE_PATH}/outliers.png"
        )

        plt.close()

    ########################################################

    def summary(self):

        print("=" * 70)

        print("Dataset Shape")

        print(self.df.shape)

        print("=" * 70)

        print(self.df.describe())

        print("=" * 70)

        print("Missing Values")

        print(self.df.isnull().sum())

    ########################################################

    def run(self):

        self.create_target()

        self.summary()

        self.duration_distribution()

        self.log_distribution()

        self.duration_by_hour()

        self.duration_by_day()

        self.passenger_count()

        self.correlation_matrix()

        self.outliers()

        self.missing_values()

        print("\nEDA Completed Successfully.")