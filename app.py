import streamlit as st
import joblib
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

