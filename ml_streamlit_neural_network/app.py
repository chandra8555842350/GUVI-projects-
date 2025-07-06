import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime,time

# Load pre-trained model
model = joblib.load("neural_network_model.pkl") 
st.title(" Energy Usage Predictor (Manual Input)")

# User Inputs 
st.subheader(" Input Features")

# Time input

# Select Date
selected_date = st.date_input(" Select Date", value=datetime.now().date())

# Select Time with 24-hour format and minute precision
selected_time = st.time_input(" Select Time (24-hour)", value= time(12, 0), step=60)

# Combine date and time into a single datetime
timestamp = datetime.combine(selected_date, selected_time)

# Display the result
st.write(" Selected Timestamp:", timestamp)




# Numerical inputs
voltage = st.number_input("Voltage" )
global_intensity = st.number_input("Global Intensity" )
global_reactive_power = st.number_input("Global Reactive Power")
sub_metering_1 = st.number_input("Sub Metering 1")
sub_metering_2 = st.number_input("Sub Metering 2")
sub_metering_3 = st.number_input("Sub Metering 3")

#  Derive Time Features
hour = timestamp.hour
day = timestamp.day
month = timestamp.month
weekday = timestamp.weekday()

sin_of_month = np.sin(2 * np.pi * month / 12)
cos_of_month = np.cos(2 * np.pi * month / 12)
sin_of_day = np.sin(2 * np.pi * day / 31)
cos_of_day = np.cos(2 * np.pi * day / 31)
sin_of_hour = np.sin(2 * np.pi * hour / 24)
cos_of_hour = np.cos(2 * np.pi * hour / 24)
is_weekend = int(weekday >= 5)
is_morning_peak = int(6 <= hour <= 10)
is_evening_peak = int(17 <= hour <= 21)

#  Prepare input data for prediction
input_data = pd.DataFrame([{
    "Voltage": voltage,
    "Global_intensity": global_intensity,
    "Global_reactive_power": global_reactive_power,
    "Sub_metering_1": sub_metering_1,
    "Sub_metering_2": sub_metering_2,
    "Sub_metering_3": sub_metering_3,
    "sin_of_month": sin_of_month,
    "cos_of_month": cos_of_month,
    "sin_of_hour": sin_of_hour,
    "cos_of_hour": cos_of_hour,
    "sin_of_day": sin_of_day,
    "cos_of_day": cos_of_day,
    "Is_Weekend": is_weekend,
    "Is_Morning_Peak": is_morning_peak,
    "Is_Evening_Peak": is_evening_peak,
}])

#  Predict 
if st.button(" Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f" **Predicted Output:** {prediction:.4f}")
