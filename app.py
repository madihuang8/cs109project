import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime


# Title and Introduction
st.title("Prostate Cancer Active Surveillance Prediction Model")
st.write("""
Welcome! I'm Madi and this demo is part of my CS109 project, which builds on research I conducted this summer at the Canary Center at Stanford. I developed predictive models for prostate cancer using logistic regression to support clinical decision-making during active surveillance. Here’s a bit of background before you dive into the demo!
""")

# Background Section
st.header("Background")
st.write("""
Prostate cancer is one of the most common cancers among older men. However, it can be challenging to distinguish between aggressive and indolent forms of the disease. Many men live with prostate cancer for years without experiencing any health issues. For these individuals, immediate treatment may not be necessary and could cause harmful side effects.
""")

# Active Surveillance Section
st.subheader("What is Active Surveillance?")
st.write("""
Active surveillance (AS) is an alternative to immediate treatment. It involves closely monitoring key indicators over time to identify if and when the cancer progresses. By doing so, unnecessary treatment can be avoided while ensuring timely intervention for those whose cancer becomes more aggressive.
""")

# Logistic Regression Model Parameters Section
st.subheader("Parameters for the Logistic Regression Model")
st.write("""
The following parameters are included in the logistic regression model to predict Active Surveillance outcomes:
- **`time_observed_days`**: How long the patient has been on active surveillance
- **`diagnostic_PSA`**: Prostate-Specific Antigen (PSA) level at diagnosis
- **`Gleason_Grade`**: Gleason Grade Group, which indicates the severity of the cancer
- **`size_prostate`**: Measurement of prostate volume
""")

# Outcomes Section
st.subheader("Active Surveillance Outcomes")
st.write("""
The model predicts one of two outcomes:
- **Censored**: Completed 5 years on AS without cancer progression
- **Reclassified**: Removed from AS protocol due to cancer progression
""")

# Vision Section
st.subheader("Project Vision")
st.write("""
The vision for this project is to create a tool that doctors and patients can use to interpret their results. When they go in to get another blood test, biopsy, or volume measurement, this model can help interpret new information in the context of the patient’s medical history!
""")

# Example Inputs
st.subheader("Example Inputs")
st.write("""
Here are some examples from the test set you can try inputting (these numbers are from real patient data!):

- **Patient 103**:
    - Measurement Set 1:
        - Age: 50
        - Date: 12/01/23
        - PSA: 6.89
        - Volume: 70
        - Gleason Grade: 1
    - Measurement Set 2:
        - Age: 55
        - Date: 12/01/24
        - PSA: 9.4
        - Volume: 70
        - Gleason Grade: 1
    - **Outcome**: Censored

- **Patient 285**:
    - Measurement Set 1:
        - Age: 60
        - Date: 6/20/23
        - PSA: 6.1
        - Volume: 144.0
        - Gleason Grade: 1
    - Measurement Set 2:
        - Age: 61
        - Date: 12/20/23
        - PSA: 5.4
        - Volume: 144.0
        - Gleason Grade: 1
    - **Outcome**: Reclassified
""")

# Notes Section
st.subheader("Notes")
st.write("""
- Make sure to click **Predict** at the bottom of the page before inputting another set of data!
- When changing the year in the date input, ensure you click the day to save the changes.
- Refresh to restart!
""")

import pickle
import os
import streamlit as st

model_path = "logistic_regression_model.pkl"

#try:
#    with open(model_path, "rb") as file:
#        model = pickle.load(file)
#    st.write("Model loaded successfully!")
#except FileNotFoundError:
#    st.error(f"Model file '{model_path}' not found. Please check the file location.")
#except Exception as e:
#    st.error(f"An error occurred while loading the model: {e}")
# Path to the logistic regression model

model_path = '/Users/madisonhuang/Documents/crest/data/logistic_regression_model.pkl'

# Load the logistic regression model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.write("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Title and description
st.title("Prostate Cancer Progression Tracker")
st.write("Track and predict cancer progression based on longitudinal measurements.")

# Initialize session state to store longitudinal data
if "measurements" not in st.session_state:
    st.session_state.measurements = {"Date": [], "PSA Level": [], "Gland Volume": [], "Gleason Score": []}

# Input widgets for static attributes
st.header("Patient Information")
age = st.number_input("Age", min_value=18, max_value=120, step=1, value=50)

# Add a new set of measurements
st.header("Add New Measurements")
new_date = st.date_input("Enter Date of Measurement:", value=datetime.today().date())
new_psa = st.number_input("Enter PSA Level (ng/mL):", min_value=0.0, max_value=50.0, step=0.1)
new_gland_volume = st.number_input("Enter Gland Volume (cm³):", min_value=0.0, max_value=200.0, step=0.1)
new_gleason_score = st.slider("Enter Gleason Grade Group:", min_value=1, max_value=5, step=1)

if st.button("Add Measurement Set"):
    # Calculate the difference in days from the previous measurement
    if st.session_state.measurements["Date"]:
        previous_date = datetime.strptime(st.session_state.measurements["Date"][-1], "%Y-%m-%d").date()
        days_since_previous = (new_date - previous_date).days
        st.write(f"Days since previous measurement: {days_since_previous}")
    else:
        days_since_previous = 0
        st.write("This is the first measurement.")

    # Add the complete set of measurements
    st.session_state.measurements["Date"].append(new_date.strftime("%Y-%m-%d"))
    st.session_state.measurements["PSA Level"].append(new_psa)
    st.session_state.measurements["Gland Volume"].append(new_gland_volume)
    st.session_state.measurements["Gleason Score"].append(new_gleason_score)
    st.success("Added new measurement set!")

# Display longitudinal data
st.header("Longitudinal Data")
data = pd.DataFrame(st.session_state.measurements)
if not data.empty:
    st.dataframe(data)

# Visualization: Plot PSA Levels and Gland Volumes over time
st.header("Measurement Trends")
if not data.empty:
    psa_data = data[["Date", "PSA Level"]].copy()
    psa_data["Date"] = pd.to_datetime(psa_data["Date"])
    st.line_chart(psa_data.set_index("Date")["PSA Level"], use_container_width=True)

    gland_data = data[["Date", "Gland Volume"]].copy()
    gland_data["Date"] = pd.to_datetime(gland_data["Date"])
    st.line_chart(gland_data.set_index("Date")["Gland Volume"], use_container_width=True)

# Predict based on the latest values
st.header("Prediction")
if len(st.session_state.measurements["PSA Level"]) > 0:
    latest_psa = st.session_state.measurements["PSA Level"][-1]
    latest_gland_volume = st.session_state.measurements["Gland Volume"][-1]
    latest_gleason_score = st.session_state.measurements["Gleason Score"][-1]
    
    # Calculate "time observed" in days
    first_date = datetime.strptime(st.session_state.measurements["Date"][0], "%Y-%m-%d")
    latest_date = datetime.strptime(st.session_state.measurements["Date"][-1], "%Y-%m-%d")
    time_observed = (latest_date - first_date).days
    
    # Features for prediction (excluding age, including time observed)
    features = np.array([[latest_psa, latest_gland_volume, latest_gleason_score, time_observed]])
    
    if st.button("Predict Progression"):
        try:
            probability = model.predict_proba(features)[0][1]  # Probability of progression
            if probability > 0.5:
                st.write(f"Prediction: Cancer is likely to progress in the next five years.")
            else:
                st.write(f"Prediction: Cancer is unlikely to progress in the next five years.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.write("Please add at least one measurement set to make a prediction.")