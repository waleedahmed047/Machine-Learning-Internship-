"""
Week 02 Internship Project

NYC Taxi Trip Duration Prediction
Streamlit Application

"""

import warnings

warnings.filterwarnings("ignore")


import streamlit as st

import pandas as pd

import numpy as np

import joblib

import os


from src.features.build_features import FeatureBuilder



##########################################################
# Page Configuration
##########################################################

st.set_page_config(

    page_title="NYC Taxi Trip Duration Predictor",

    page_icon="🚕",

    layout="wide"

)



##########################################################
# Title
##########################################################

st.title(

    "🚕 NYC Taxi Trip Duration Prediction"

)


st.write(

    """

    This application predicts NYC taxi trip duration

    using a trained Machine Learning model.

    """

)



##########################################################
# Load Model
##########################################################

@st.cache_resource

def load_model():


    model_path = (

        "models/tuned_random_forest.joblib"

    )


    if os.path.exists(model_path):

        model = joblib.load(

            model_path

        )

        return model


    else:

        st.error(

            "Model file not found!"

        )

        return None



model = load_model()



##########################################################
# Load Feature Builder
##########################################################

builder = FeatureBuilder()



##########################################################
# Sidebar Information
##########################################################

st.sidebar.header(

    "About Project"

)


st.sidebar.info(

    """

    Dataset:

    NYC Taxi Trip Duration Dataset


    Model:

    Tuned Random Forest Regressor


    Task:

    Regression Prediction


    """

)

##########################################################
# User Input Section
##########################################################

st.header(
    "Enter Trip Details"
)


col1, col2 = st.columns(2)


with col1:

    vendor_id = st.selectbox(

        "Vendor ID",

        [1, 2]

    )


    passenger_count = st.number_input(

        "Passenger Count",

        min_value=1,

        max_value=8,

        value=1

    )


    pickup_longitude = st.number_input(

        "Pickup Longitude",

        value=-73.9857,

        format="%.6f"

    )


    pickup_latitude = st.number_input(

        "Pickup Latitude",

        value=40.7484,

        format="%.6f"

    )



with col2:

    dropoff_longitude = st.number_input(

        "Dropoff Longitude",

        value=-73.9851,

        format="%.6f"

    )


    dropoff_latitude = st.number_input(

        "Dropoff Latitude",

        value=40.7580,

        format="%.6f"

    )


    store_and_fwd_flag = st.selectbox(

        "Store and Forward Flag",

        ["N", "Y"]

    )


    pickup_datetime = st.date_input(

        "Pickup Date"

    )



##########################################################
# Time Input
##########################################################

pickup_time = st.time_input(

    "Pickup Time"

)


##########################################################
# Create Prediction Data
##########################################################

def create_input_data():


    datetime_value = pd.to_datetime(

        str(pickup_datetime)

        + " "

        + str(pickup_time)

    )


    input_df = pd.DataFrame({

        "vendor_id":

            [vendor_id],


        "pickup_datetime":

            [datetime_value],


        "dropoff_datetime":

            [datetime_value],


        "passenger_count":

            [passenger_count],


        "pickup_longitude":

            [pickup_longitude],


        "pickup_latitude":

            [pickup_latitude],


        "dropoff_longitude":

            [dropoff_longitude],


        "dropoff_latitude":

            [dropoff_latitude],


        "store_and_fwd_flag":

            [store_and_fwd_flag]

    })


    return input_df



##########################################################
# Prediction Button
##########################################################

predict_button = st.button(

    "🚕 Predict Trip Duration"

)


if predict_button:


    input_data = create_input_data()


    st.subheader(

        "Input Data"

    )


    st.write(

        input_data

    )


    try:


        prediction = model.predict(

            input_data

        )


        duration = prediction[0]


        st.success(

            f"Predicted Trip Duration: {duration:.2f} seconds"

        )


        minutes = duration / 60


        st.info(

            f"Approximately {minutes:.2f} minutes"

        )


    except Exception as e:


        st.error(

            f"Prediction Error: {e}"

        )
        ##########################################################
# Model Information Section
##########################################################

st.divider()


st.header(
    "📊 Model Information"
)


info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.metric(

        label="Algorithm",

        value="Random Forest"

    )


with info_col2:

    st.metric(

        label="Task",

        value="Regression"

    )


with info_col3:

    st.metric(

        label="Prediction",

        value="Trip Duration"

    )



##########################################################
# Feature Information
##########################################################

with st.expander(
    "ℹ️ Input Feature Explanation"
):

    st.write(

        """

        **Vendor ID**

        Taxi company identifier.


        **Passenger Count**

        Number of passengers in the taxi.


        **Pickup Coordinates**

        Starting location of the trip.


        **Dropoff Coordinates**

        Destination location.


        **Store and Forward Flag**

        Indicates if trip data was stored before sending.


        **Pickup Date and Time**

        Used for extracting time-based features.

        """

    )



##########################################################
# Prediction History
##########################################################

if "history" not in st.session_state:

    st.session_state.history = []



if predict_button:

    try:

        history_record = {

            "Passenger":

                passenger_count,


            "Prediction(seconds)":

                round(duration, 2),


            "Prediction(minutes)":

                round(minutes, 2)

        }


        st.session_state.history.append(

            history_record

        )


    except:

        pass



##########################################################
# Display History
##########################################################

if len(st.session_state.history) > 0:


    st.divider()


    st.header(

        "🕒 Prediction History"

    )


    history_df = pd.DataFrame(

        st.session_state.history

    )


    st.dataframe(

        history_df,

        use_container_width=True

    )



##########################################################
# Footer
##########################################################

st.divider()


st.caption(

    """

    NYC Taxi Trip Duration Prediction System

    Built using Machine Learning

    Model: Tuned Random Forest Regressor

    """

)