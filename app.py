import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load saved files
# -------------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Fraud Detection App", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and check if it is Fraud or Not Fraud.")

# Show expected feature count (debug)
st.info(f"Scaler expects {scaler.n_features_in_} features")

# -------------------------------
# Inputs
# -------------------------------
# (Optional) If scaler expects 9 features, we will add ID automatically
id_val = st.number_input("Transaction ID (optional)", min_value=1, value=1)

amount = st.number_input("Amount", min_value=0.0, value=100.0)
transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)

merchant_category = st.selectbox("Merchant Category", list(le.classes_))

foreign_transaction = st.selectbox("Foreign Transaction", [0, 1])
location_mismatch = st.selectbox("Location Mismatch", [0, 1])

device_trust_score = st.slider("Device Trust Score", 0, 100, 50)
velocity_last_24h = st.number_input("Velocity Last 24 Hours", min_value=0, value=1)
cardholder_age = st.number_input("Cardholder Age", min_value=1, value=30)

# Encode merchant category
merchant_encoded = le.transform([merchant_category])[0]

# -------------------------------
# Create input data (8 base features)
# -------------------------------
features_dict = {
    "amount": amount,
    "transaction_hour": transaction_hour,
    "merchant_category": merchant_encoded,
    "foreign_transaction": foreign_transaction,
    "location_mismatch": location_mismatch,
    "device_trust_score": device_trust_score,
    "velocity_last_24h": velocity_last_24h,
    "cardholder_age": cardholder_age
}

# Convert to DataFrame
input_data = pd.DataFrame([features_dict])

# -------------------------------
# Fix feature mismatch automatically
# -------------------------------
expected_features = scaler.n_features_in_
current_features = input_data.shape[1]

# If scaler expects 9 features, add id column
if expected_features == 9 and current_features == 8:
    input_data.insert(0, "id", id_val)

# If still mismatch, stop and show error
if input_data.shape[1] != expected_features:
    st.error(
        f"❌ Feature mismatch!\n\n"
        f"Scaler expects {expected_features} features, but app is sending {input_data.shape[1]} features.\n\n"
        f"Your training used different columns. Please retrain model or tell me the feature columns."
    )
    st.write("Current input columns:", list(input_data.columns))
    st.stop()

# -------------------------------
# Scale + Predict
# -------------------------------
scaled_data = scaler.transform(input_data.values)

if st.button("Check Fraud 🚨"):
    pred = model.predict(scaled_data)[0]

    # Some models may not support predict_proba
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_data)[0][1]
        prob_text = f"{prob:.2f}"
    else:
        prob_text = "N/A"

    if pred == 1:
        st.error(f"🚨 FRAUD Transaction Detected! (Probability: {prob_text})")
    else:
        st.success(f"✅ NOT FRAUD Transaction (Probability: {prob_text})")
