import streamlit as st
import pickle
import numpy as np

# Load the KMeans model
with open("kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Customer Segmentation", layout="centered", page_icon=":bar_chart:")
st.title("Customer Segmentation — KMeans Predictor")

st.markdown("Enter the feature values below. The form is built from your features, so it matches training exactly.")

# Input fields
income = st.number_input("Income", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
spending = st.number_input("Total_Spending", min_value=0.0, step=100.0, value=1000.0, format="%.2f")

# Predict button
if st.button("Predict segment"):
    features = np.array([[income, spending]])
    segment = kmeans.predict(features)[0]
    
    st.success(f"Predicted Segment: {segment}")
