import streamlit as st
import joblib
import pandas as pd

# Load saved files
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Fraud Detection App", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and check if it is Fraud or Not Fraud.")

# Inputs
amount = st.number_input("Amount", min_value=0.0, value=100.0)
transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)

merchant_category = st.selectbox(
    "Merchant Category",
    ["Electronics", "Travel", "Grocery", "Food"]
)

foreign_transaction = st.selectbox("Foreign Transaction", [0, 1])
location_mismatch = st.selectbox("Location Mismatch", [0, 1])

device_trust_score = st.slider("Device Trust Score", 0, 100, 50)
velocity_last_24h = st.number_input("Velocity Last 24 Hours", min_value=0, value=1)
cardholder_age = st.number_input("Cardholder Age", min_value=1, value=30)

# Encode merchant category
merchant_encoded = le.transform([merchant_category])[0]

# Create dataframe (must match training feature order)
input_data = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "merchant_category": merchant_encoded,
    "foreign_transaction": foreign_transaction,
    "location_mismatch": location_mismatch,
    "device_trust_score": device_trust_score,
    "velocity_last_24h": velocity_last_24h,
    "cardholder_age": cardholder_age
}])

# Scale numeric columns
scaled_data = scaler.transform(input_data.values)

# Predict
if st.button("Check Fraud 🚨"):
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if pred == 1:
        st.error(f"🚨 FRAUD Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ NOT FRAUD Transaction (Probability: {prob:.2f})")
