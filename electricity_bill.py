```python
import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Electricity Bill Prediction",
    page_icon="⚡",
    layout="centered"
)

# Load trained model
model = joblib.load("PolynomialRegression_ElectricBill_AC.pkl")

# Application title
st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict the Electric Bill using AC electricity consumption "
    "with Polynomial Regression."
)

st.divider()

# AC Units input
ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=10.0,
    step=1.0
)

# Prediction button
if st.button("Predict Electric Bill"):

    # Prepare input
    input_data = np.array([[ac_units]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(
        f"Predicted Electric Bill: ₹{prediction:.2f}"
    )

    st.write(f"AC Units: **{ac_units:.2f}**")
```
