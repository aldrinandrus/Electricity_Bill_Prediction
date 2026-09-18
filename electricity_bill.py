
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Electricity Bill Prediction",
    page_icon="⚡",
    layout="centered"
)

# ============================================================
# Load Model
# ============================================================

MODEL_FILE = "PolynomialRegression_ElectricBill_AC.pkl"

model = joblib.load(MODEL_FILE)

# ============================================================
# Application Title
# ============================================================

st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict the Electric Bill using AC electricity "
    "consumption with Polynomial Regression."
)

st.divider()

# ============================================================
# User Input
# ============================================================

ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=10.0,
    step=1.0
)

# ============================================================
# Prediction
# ============================================================

if st.button("Predict Electric Bill"):

    # Original input
    input_data = np.array([[ac_units]])

    # --------------------------------------------------------
    # Create Polynomial Features
    # Degree = 2
    # --------------------------------------------------------

    polynomial = PolynomialFeatures(
        degree=2,
        include_bias=True
    )

    polynomial_input = polynomial.fit_transform(input_data)

    # --------------------------------------------------------
    # Check number of features expected by the model
    # --------------------------------------------------------

    expected_features = model.n_features_in_
    actual_features = polynomial_input.shape[1]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if expected_features == actual_features:

        prediction = model.predict(polynomial_input)[0]

        st.success(
            f"Predicted Electric Bill: ₹{prediction:.2f}"
        )

        st.write(
            f"AC Units: **{ac_units:.2f}**"
        )

    else:

        st.error(
            f"Model expects {expected_features} features, "
            f"but the polynomial transformation produced "
            f"{actual_features} features."
        )

        st.info(
            "The saved .pkl model was trained with a different "
            "feature/preprocessing configuration."
        )

