# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load models and encoder
# -----------------------------
rf_regressor = joblib.load("rf_model.pkl")       # Regression model
clf_model = joblib.load("xgb_clf.pkl") # Classification model
label_encoder = joblib.load("label_encoder.pkl")    # LabelEncoder for classification

# Automatically get feature names used during training
reg_columns = rf_regressor.feature_names_in_
clf_columns = clf_model.feature_names_in_

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Tourist Attraction Prediction App")

# -----------------------------
# User Input: Raw Features
# -----------------------------
st.header("Enter Raw Features")
raw_input = {
    'TransactionId': st.text_input("TransactionId"),
    'UserId': st.text_input("UserId"),
    'VisitYear': st.number_input("VisitYear", 2000, 2030, 2025),
    'VisitMonth': st.number_input("VisitMonth", 1, 12, 1),
    'VisitModeId': st.number_input("VisitModeId", 0, 10, 1),
    'AttractionId': st.number_input("AttractionId", 0),
    'Rating': st.number_input("Rating", 0.0, 5.0, 3.0),
    'VisitMode': st.text_input("VisitMode"),
    'AttractionCityId': st.number_input("AttractionCityId", 0),
    'AttractionTypeId': st.number_input("AttractionTypeId", 0),
    'Attraction': st.text_input("Attraction"),
    'AttractionAddress': st.text_input("AttractionAddress"),
    'AttractionType': st.text_input("AttractionType"),
    'ContinentId': st.number_input("ContinentId", 0),
    'RegionId': st.number_input("RegionId", 0),
    'CountryId': st.number_input("CountryId", 0),
    'CityId': st.number_input("CityId", 0),
    'CityName': st.text_input("CityName"),
    'Continent': st.text_input("Continent"),
    'Country': st.text_input("Country"),
    'Region': st.text_input("Region")
}

raw_df = pd.DataFrame([raw_input])

# -----------------------------
# Derivative Features
# -----------------------------
def generate_derivative_features(df):
    df['visit_day_of_year'] = pd.to_datetime(
        df['VisitYear'].astype(str) + '-' + df['VisitMonth'].astype(str) + '-01'
    ).dt.dayofyear
    
    # 2. One-hot encode categorical columns
    categorical_cols = ['AttractionType', 'CityName', 'Continent', 'Country', 'Region']
    for col in categorical_cols:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    
    return df

derivative_df = generate_derivative_features(raw_df)

# -----------------------------
# Tabs for Regression and Classification
# -----------------------------
tab1, tab2 = st.tabs(["Regression - Rating Prediction", "Classification - Visit Mode"])

# -----------------------------
# Tab 1: Regression
# -----------------------------
with tab1:
    st.subheader("Predict Rating")
    
    reg_features = derivative_df.copy()
    
    # Ensure all regression model columns exist
    for col in reg_columns:
        if col not in reg_features.columns:
            reg_features[col] = 0
    reg_features = reg_features[reg_columns]
    
    if st.button("Predict Rating"):
        reg_pred = rf_regressor.predict(reg_features)[0]
        st.success(f"Predicted Rating: {reg_pred:.2f}")

# -----------------------------
# Tab 2: Classification
# -----------------------------
with tab2:
    st.subheader("Predict Visit Mode")
    
    clf_features = derivative_df.copy()
    
    # Ensure all classification model columns exist
    for col in clf_columns:
        if col not in clf_features.columns:
            clf_features[col] = 0
    clf_features = clf_features[clf_columns]
    
    if st.button("Predict Visit Mode"):
        clf_pred = clf_model.predict(clf_features)[0]
        clf_label = label_encoder.inverse_transform([clf_pred])[0]
        st.success(f"Predicted Visit Mode: {clf_label}")
