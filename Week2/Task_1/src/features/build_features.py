"""
Week 03 Advanced Regression
Feature Engineering Module
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        ####################################################
        # Convert pickup datetime
        ####################################################

        if "pickup_datetime" in df.columns:

            df["pickup_datetime"] = pd.to_datetime(
                df["pickup_datetime"]
            )

            df["pickup_hour"] = df["pickup_datetime"].dt.hour

            df["pickup_day"] = df["pickup_datetime"].dt.day

            df["pickup_month"] = df["pickup_datetime"].dt.month

            df["pickup_dayofweek"] = (
                df["pickup_datetime"].dt.dayofweek
            )

            df["pickup_week"] = (
                df["pickup_datetime"]
                .dt.isocalendar()
                .week.astype(int)
            )

            ####################################################
            # Weekend
            ####################################################

            df["is_weekend"] = (
                df["pickup_dayofweek"] >= 5
            ).astype(int)

            ####################################################
            # Peak Hour
            ####################################################

            df["is_peak_hour"] = (
                (
                    (df["pickup_hour"] >= 7)
                    &
                    (df["pickup_hour"] <= 10)
                )
                |
                (
                    (df["pickup_hour"] >= 16)
                    &
                    (df["pickup_hour"] <= 19)
                )
            ).astype(int)

            ####################################################
            # Night Trips
            ####################################################

            df["is_night_trip"] = (
                (
                    df["pickup_hour"] <= 5
                )
                |
                (
                    df["pickup_hour"] >= 22
                )
            ).astype(int)

            ####################################################
            # Cyclic Encoding
            ####################################################

            df["hour_sin"] = np.sin(
                2 * np.pi * df["pickup_hour"] / 24
            )

            df["hour_cos"] = np.cos(
                2 * np.pi * df["pickup_hour"] / 24
            )

            df["day_sin"] = np.sin(
                2 * np.pi * df["pickup_dayofweek"] / 7
            )

            df["day_cos"] = np.cos(
                2 * np.pi * df["pickup_dayofweek"] / 7
            )

        ####################################################
        # Passenger Features
        ####################################################

        if "passenger_count" in df.columns:

            df["is_single"] = (
                df["passenger_count"] == 1
            ).astype(int)

            df["is_group"] = (
                df["passenger_count"] >= 4
            ).astype(int)

        ####################################################
        # Route Feature
        ####################################################

        if (
            "PULocationID" in df.columns
            and
            "DOLocationID" in df.columns
        ):

            df["route"] = (
                df["PULocationID"].astype(str)
                +
                "_"
                +
                df["DOLocationID"].astype(str)
            )

            df["same_zone"] = (
                df["PULocationID"]
                ==
                df["DOLocationID"]
            ).astype(int)

        ####################################################
        # Airport Features
        ####################################################

        airport_ids = [1, 132, 138]

        if "PULocationID" in df.columns:

            df["airport_pickup"] = (
                df["PULocationID"]
                .isin(airport_ids)
            ).astype(int)

        if "DOLocationID" in df.columns:

            df["airport_dropoff"] = (
                df["DOLocationID"]
                .isin(airport_ids)
            ).astype(int)

        ####################################################
        # Distance Features
        ####################################################

        if "trip_distance" in df.columns:

            df["distance_squared"] = (
                df["trip_distance"] ** 2
            )

            if "passenger_count" in df.columns:

                df["distance_per_passenger"] = (
                    df["trip_distance"]
                    /
                    (df["passenger_count"] + 1)
                )

            df["long_trip"] = (
                df["trip_distance"] > 10
            ).astype(int)

        ####################################################
        # Remove Leakage Columns
        ####################################################

        leakage_columns = [

            "dropoff_datetime",

            "fare_amount",

            "tip_amount",

            "tolls_amount",

            "total_amount",

            "extra",

            "mta_tax",

            "airport_fee",

            "improvement_surcharge",

            "congestion_surcharge"

        ]

        existing = [

            col
            for col in leakage_columns
            if col in df.columns

        ]

        df.drop(
            columns=existing,
            inplace=True,
            errors="ignore"
        )

        ####################################################
        # Remove Raw Datetime
        ####################################################

        df.drop(
            columns=["pickup_datetime"],
            inplace=True,
            errors="ignore"
        )

        return df