import streamlit as st
import joblib
import numpy as np

# Page configuration

st.set_page_config(
page_title="Electricity Bill Prediction",
page_icon="⚡",
layout="centered"
)

# Load trained polynomial regression model

model = joblib.load("PolynomialRegression_ElectricBill_AC.pkl")

# Title

st.title("⚡ Electricity Bill Prediction")

st.write(
"Predict the Electric Bill using AC electricity consumption "
"with a Polynomial Regression model."
)

st.divider()

# Input

ac_units = st.number_input(
"Enter AC Units",
min_value=0.0,
value=10.0,
step=1.0
)

# Prediction

if st.button("Predict Electric Bill"):


# Convert input into the format expected by the model
input_data = np.array([[ac_units]])

# Make prediction
prediction = model.predict(input_data)[0]

# Display result
st.success(
    f"Predicted Electric Bill: ₹{prediction:.2f}"
)

st.write(f"AC Units: **{ac_units:.2f}**")

