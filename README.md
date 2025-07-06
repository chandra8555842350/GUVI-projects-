Energy Usage Predictor (Streamlit App)
This Streamlit application predicts energy usage based on user inputs and time-based features using a trained neural network model. 

Features
Date and Time input

Manual input of electrical readings (voltage, current, sub-metering)

Time-feature engineering using sin/cos transforms

Prediction using a trained neural network model

User-friendly Streamlit interface

Project Approach
Step 1: Data Collection and Understanding
Used publicly available power consumption dataset.

Focused on key electrical parameters like Voltage, Global Intensity, Reactive Power, and Sub Metering.

Step 2: Data Preprocessing
Removed missing or duplicate entries.

Handled outliers and normalized features where needed.

Step 3: Feature Engineering
Created cyclic features using sine and cosine transforms:

sin_of_hour, cos_of_hour, sin_of_day, cos_of_day, sin_of_month, cos_of_month

Added binary flags:

Is_Weekend, Is_Morning_Peak, Is_Evening_Peak

Step 4: Model Building
Trained a neural network using scikit-learn on cleaned and engineered data.

Saved the model using joblib for deployment.

Step 5: Streamlit Deployment
Built an interactive UI using Streamlit for real-time prediction.

Inputs accepted: date, time, voltage, intensity, reactive power, sub-metering values.

Model outputs predicted energy usage.
