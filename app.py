import streamlit as st
import pickle
import numpy as np

# Load scaler, PCA, and KMeans model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation", layout="centered", page_icon=":bar_chart:")
st.title("Customer Segmentation — KMeans Predictor")

st.markdown("Enter the feature values below. The form matches your training features exactly.")

income = st.number_input("Income", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
spending = st.number_input("Total_Spending", min_value=0.0, step=100.0, value=1000.0, format="%.2f")

if st.button("Predict segment"):
    # Prepare input features
    features = np.array([[income, spending]])
    # Scale features
    features_scaled = scaler.transform(features)
    # Apply PCA
    features_pca = pca.transform(features_scaled)
    # Predict segment
    segment = kmeans.predict(features_pca)[0]
    
    st.success(f"Predicted Segment: {segment}")
